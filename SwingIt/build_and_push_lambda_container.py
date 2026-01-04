#!/usr/bin/env python3
"""
Build and push Lambda container image to Amazon ECR

Usage:
    python build_and_push_lambda_container.py

Requirements:
    AWS CLI configured with appropriate credentials
    Docker installed and running
    Appropriate IAM permissions for ECR
"""

import subprocess
import sys
import os
import json
from datetime import datetime

# Configuration
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
ECR_REPOSITORY_NAME = os.environ.get('ECR_REPOSITORY_NAME', 'forehand-analysis-lambda')
LAMBDA_FUNCTION_NAME = os.environ.get('LAMBDA_FUNCTION_NAME', 'ForehandAnalysis')
DOCKERFILE = 'Dockerfile.lambda.container'
IMAGE_TAG = 'latest'

def run_command(cmd, check=True, capture_output=False):
    """Run a shell command and return the result"""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        if capture_output:
            result = subprocess.run(cmd, shell=isinstance(cmd, str), check=check, 
                                  capture_output=True, text=True)
            return result.stdout.strip()
        else:
            subprocess.run(cmd, shell=isinstance(cmd, str), check=check)
            return None
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        if capture_output:
            print(f"stdout: {e.stdout}")
            print(f"stderr: {e.stderr}")
        sys.exit(1)

def get_aws_account_id():
    """Get AWS account ID"""
    print("Getting AWS account ID...")
    try:
        account_id = run_command(
            ['aws', 'sts', 'get-caller-identity', '--query', 'Account', '--output', 'text', '--region', AWS_REGION],
            capture_output=True
        )
        print(f"AWS Account ID: {account_id}")
        return account_id
    except Exception as e:
        print(f"Failed to get AWS account ID: {e}")
        sys.exit(1)

def create_ecr_repository(repo_name):
    """Create ECR repository if it doesn't exist"""
    print(f"Checking if ECR repository '{repo_name}' exists...")
    result = subprocess.run(
        ['aws', 'ecr', 'describe-repositories', '--repository-names', repo_name, '--region', AWS_REGION],
        capture_output=True,
        text=True
    )
    if result.returncode == 0:
        print(f"Repository '{repo_name}' already exists")
    else:
        print(f"Creating ECR repository '{repo_name}'...")
        run_command([
            'aws', 'ecr', 'create-repository',
            '--repository-name', repo_name,
            '--region', AWS_REGION,
            '--image-scanning-configuration', 'scanOnPush=true'
        ])
        print(f"Repository '{repo_name}' created successfully")

def get_ecr_login_command():
    """Get ECR login command"""
    print("Getting ECR login token...")
    login_cmd = run_command(
        ['aws', 'ecr', 'get-login-password', '--region', AWS_REGION],
        capture_output=True
    )
    return login_cmd

def docker_login(ecr_uri, login_token):
    """Login to ECR"""
    print(f"Logging into ECR: {ecr_uri}")
    try:
        process = subprocess.Popen(
            ['docker', 'login', '--username', 'AWS', '--password-stdin', ecr_uri],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        stdout, stderr = process.communicate(input=login_token)
        if process.returncode != 0:
            print(f"Error logging into ECR: {stderr}")
            sys.exit(1)
        print("Logged into ECR successfully")
    except Exception as e:
        print(f"Error logging into ECR: {e}")
        sys.exit(1)

def build_docker_image(ecr_uri, tag):
    """Build Docker image for Lambda compatibility"""
    image_uri = f"{ecr_uri}:{tag}"
    local_tag = f"forehand-analysis-lambda-local:{tag}"
    print(f"Building Docker image: {image_uri}")
    
    # Use buildx with --load to force single-platform image
    try:
        result = subprocess.run(['docker', 'buildx', 'version'], 
                              capture_output=True, text=True, check=False)
        if result.returncode == 0:
            # Create builder if needed
            try:
                subprocess.run(['docker', 'buildx', 'create', '--name', 'lambda-builder', '--use', '--driver', 'docker-container'], 
                              check=False, capture_output=True)
            except:
                pass
            
            # Build and load locally
            run_command([
                'docker', 'buildx', 'build',
                '--platform', 'linux/amd64',
                '--load',  # Load into local Docker daemon
                '-f', DOCKERFILE,
                '-t', local_tag,
                '.'
            ])
            
            # Tag for ECR
            run_command(['docker', 'tag', local_tag, image_uri])
            print(f"Image built and tagged successfully: {image_uri}")
            return image_uri
    except Exception as e:
        print(f"buildx approach failed: {e}, trying standard build...")
    
    # Fallback to standard docker build
    print("Using standard docker build...")
    run_command([
        'docker', 'build',
        '--platform', 'linux/amd64',
        '-f', DOCKERFILE,
        '-t', image_uri,
        '.'
    ])
    print(f"Image built successfully: {image_uri}")
    return image_uri

def push_docker_image(image_uri):
    """Push Docker image to ECR"""
    print(f"Pushing image to ECR: {image_uri}")
    run_command(['docker', 'push', image_uri])
    print(f"Image pushed successfully: {image_uri}")

def main():
    print("=" * 60)
    print("AWS Lambda Container Image Builder")
    print("=" * 60)
    print(f"Region: {AWS_REGION}")
    print(f"ECR Repository: {ECR_REPOSITORY_NAME}")
    print(f"Lambda Function: {LAMBDA_FUNCTION_NAME}")
    print(f"Dockerfile: {DOCKERFILE}")
    print("=" * 60)
    print()

    # Check if Dockerfile exists
    if not os.path.exists(DOCKERFILE):
        print(f"Error: Dockerfile '{DOCKERFILE}' not found")
        sys.exit(1)

    # Check if requirements_lambda.txt exists
    if not os.path.exists('requirements_lambda.txt'):
        print("Error: requirements_lambda.txt not found")
        sys.exit(1)

    # Check if main.py exists
    if not os.path.exists('main.py'):
        print("Error: main.py not found")
        sys.exit(1)

    # Get AWS account ID
    account_id = get_aws_account_id()
    ecr_uri = f"{account_id}.dkr.ecr.{AWS_REGION}.amazonaws.com/{ECR_REPOSITORY_NAME}"

    # Create ECR repository if needed
    create_ecr_repository(ECR_REPOSITORY_NAME)

    # Get ECR login token and login
    login_token = get_ecr_login_command()
    docker_login(ecr_uri, login_token)

    # Build Docker image
    image_uri = build_docker_image(ecr_uri, IMAGE_TAG)

    # Push image to ECR
    push_docker_image(image_uri)

    print()
    print("=" * 60)
    print("Success! Container image built and pushed to ECR")
    print("=" * 60)
    print(f"Image URI: {image_uri}")
    print("=" * 60)

if __name__ == '__main__':
    main()

