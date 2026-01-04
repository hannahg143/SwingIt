import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, Dimensions, Alert, ScrollView, TouchableOpacity, Image, ActivityIndicator, Platform } from 'react-native';
import { useState, useRef } from 'react';
import { Video, ResizeMode } from 'expo-av';
import * as DocumentPicker from 'expo-document-picker';
import AWS from 'aws-sdk';

// AWS Configurations
const AWS_REGION = 'us-east-1'; 
const S3_BUCKET_NAME = 'swinglt-us-east-1'; // S3 bucket name
const LAMBDA_FUNCTION_NAME = 'ForehandAnalysis'; // Lambda function name
const PRESIGNED_URL_LAMBDA_URL = ''; 
const LAMBDA_FUNCTION_URL = ''; 

const AWS_ACCESS_KEY_ID = ''; // Add AWS Access Key ID here
const AWS_SECRET_ACCESS_KEY = ''; //Add AWS Secret Access Key here

// Configure AWS SDK
AWS.config.update({
  region: AWS_REGION,
  accessKeyId: AWS_ACCESS_KEY_ID,
  secretAccessKey: AWS_SECRET_ACCESS_KEY
});

const s3 = new AWS.S3({ region: AWS_REGION });
const lambda = new AWS.Lambda({ region: AWS_REGION });

const windowWidth = Dimensions.get('window').width;
const windowHeight = Dimensions.get('window').height;

const isMobile = windowWidth < 600;
const VIDEO_WIDTH = isMobile ? windowWidth - 32 : Math.min(windowWidth * 0.95, 900);
const VIDEO_HEIGHT = isMobile ? Math.min((windowWidth - 32) * 16 / 9, windowHeight * 0.5) : 500;

export default function App() {
  const [selectedVideo, setSelectedVideo] = useState(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [analysisResults, setAnalysisResults] = useState(null);
  const [error, setError] = useState(null);
  const [identifier, setIdentifier] = useState(null);
  const videoRef = useRef(null);

  const pickVideo = async () => {
    try {
      setError(null);
      setAnalysisResults(null);
      const result = await DocumentPicker.getDocumentAsync({
        type: 'video/*',
        copyToCacheDirectory: true,
      });
      if (!result.canceled && result.assets && result.assets.length > 0) {
        const video = result.assets[0];
        setSelectedVideo(video);
        setIsPlaying(false);
      }
    } catch (error) {
      setError('Failed to pick video file');
      console.error('Video error:', error);
    }
  };

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const pollForFilesByIdentifier = async (id, timeoutMs = 120000, intervalMs = 1500) => {
    const expected = {
      backswing: '${id}1',
      midswing: '${id}2',
      followthrough: '${id}3',
    };
    const files = { backswing: null, midswing: null, followthrough: null };
    const deadline = Date.now() + timeoutMs;
    let pollCount = 0;

    while (Date.now() < deadline) {
      pollCount++;
      for (const key of Object.keys(expected)) {
        if (!files[key]) {
          try {
            // Check if file exists in S3
            await s3.headObject({
              Bucket: S3_BUCKET_NAME,
              Key: expected[key]
            }).promise();
            
            console.log('File found: ${expected[key]}');
            
            // Generate URL for the file
            const url = s3.getSignedUrl('getObject', {
              Bucket: S3_BUCKET_NAME,
              Key: expected[key],
              Expires: 3600 
            });
            
            if (!url || !url.includes('?')) {
              console.error('Invalid presigned URL generated for ${expected[key]}: ${url}');
              throw new Error('Invalid presigned URL generated');
            }
            
            console.log('Presigned URL generated for ${key}: ${url.substring(0, 100)}...');
            
            files[key] = {
              key: expected[key],
              url: url
            };
          } catch (error) {
            if (error.code === 'NotFound' || error.statusCode === 404) {
              // File doesn't exist yet, continue polling (this is expected)
              if (pollCount % 5 === 0) {
                console.log('Still waiting for ${key} (${expected[key]})... (poll #${pollCount}, ${Math.round((deadline - Date.now()) / 1000)}s remaining)');
              }
            } else {
              // Other errors
              console.error('Error checking ${key} (${expected[key]}):', error.code || error.message, error);
            }
          }
        }
      }
      if (files.backswing && files.midswing && files.followthrough) {
        console.log('All files found!', files);
        break;
      }
      await wait(intervalMs);
    }

    const foundCount = Object.values(files).filter(f => f !== null).length;
    console.log('Polling complete, found ${foundCount}/3 files');

    return files;
  };

  const handlePlayPause = async () => {
    if (videoRef.current) {
      if (isPlaying) {
        await videoRef.current.pauseAsync();
      } else {
        await videoRef.current.playAsync();
      }
      setIsPlaying(!isPlaying);
    }
  };

  const handleVideoStatusUpdate = (status) => {
    setIsPlaying(status.isPlaying);
  };


  const uploadForehand = async () => {
    if (!selectedVideo) {
      setError('Please select a video first.');
      return;
    }

    setIsUploading(true);
    setError('');

    try {
      const videoKey = "videos/${Date.now()}_${selectedVideo.name || 'video.mp4'}";
      
      // Get video blob for S3 upload
      let videoBlob;
      if (Platform.OS === 'web') {
        const response = await fetch(selectedVideo.uri);
        videoBlob = await response.blob();
      } else {
        const response = await fetch(selectedVideo.uri);
        videoBlob = await response.blob();
      }

      let presignedUrl;
      
      // Try using Lambda Function URL first (if configured), otherwise use SDK
      if (PRESIGNED_URL_LAMBDA_URL) {
        try {
          const response = await fetch(PRESIGNED_URL_LAMBDA_URL, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              action: 'generateUploadUrl',
              bucket: S3_BUCKET_NAME,
              key: videoKey,
              contentType: 'video/mp4'
            })
          });
          const data = await response.json();
          if (data.success) {
            presignedUrl = data.url;
            console.log('Presigned URL generated via Lambda Function URL');
          } else {
            throw new Error(data.error || 'Failed to generate presigned URL');
          }
        } catch (lambdaError) {
          console.warn('Lambda Function URL failed, trying SDK:', lambdaError);
          // Fall back to SDK
          try {
            presignedUrl = s3.getSignedUrl('putObject', {
              Bucket: S3_BUCKET_NAME,
              Key: videoKey,
              ContentType: 'video/mp4',
              Expires: 3600
            });
            console.log('Presigned URL generated via SDK');
          } catch (urlError) {
            throw new Error('Both Lambda URL and SDK failed, SDK error: ${urlError.message}');
          }
        }
      } else {
        // Use SDK directly
        try {
          presignedUrl = s3.getSignedUrl('putObject', {
            Bucket: S3_BUCKET_NAME,
            Key: videoKey,
            ContentType: 'video/mp4',
            Expires: 3600 // 1 hour
          });
          console.log('Presigned URL generated via SDK');
        } catch (urlError) {
          console.error('Failed to generate presigned URL:', urlError);
          throw new Error('Failed to generate presigned URL: ${urlError.message}');
        }
      }

      // Upload video to S3 using presigned URL
      console.log('Uploading video to s3://${S3_BUCKET_NAME}/${videoKey}');
      let uploadResponse;
      try {
        uploadResponse = await fetch(presignedUrl, {
          method: 'PUT',
          body: videoBlob,
          headers: {
            'Content-Type': 'video/mp4'
          }
        });
      } catch (fetchError) {
        console.error('Fetch error details:', fetchError);
        throw new Error('Failed to upload to S3: ${fetchError.message}');
      }

      if (!uploadResponse.ok) {
        const errorText = await uploadResponse.text().catch(() => 'No error details');
        console.error('S3 upload failed:', uploadResponse.status, uploadResponse.statusText, errorText);
        throw new Error('S3 upload failed: ${uploadResponse.status} ${uploadResponse.statusText}. ${errorText}.');
      }

      console.log('Video uploaded successfully');

      // Generate unique identifier for output files
      const id = String(Date.now());
      setIdentifier(id);

      // Invoke Lambda function using AWS SDK
      const lambdaPayload = {
        fileId: videoKey,
        userId: 'user',
        backswingId: '${id}1',
        midswingId: '${id}2',
        followthroughId: '${id}3'
      };

      console.log('Invoking Lambda function:', lambdaPayload);
      
      // Invoke Lambda function: use Function URL if available, otherwise try SDK
      let lambdaInvoked = false;
      
      if (LAMBDA_FUNCTION_URL) {
        // Use Lambda Function URL
        try {
          console.log('Invoking Lambda via Function URL:', LAMBDA_FUNCTION_URL);
          const response = await fetch(LAMBDA_FUNCTION_URL, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json'
            },
            body: JSON.stringify(lambdaPayload)
          });
          
          if (response.ok) {
            console.log('Lambda function invoked successfully via Function URL');
            lambdaInvoked = true;
          } else {
            const errorText = await response.text();
            throw new Error('Lambda Function URL returned ${response.status}: ${errorText}');
          }
        } catch (urlError) {
          console.error('Lambda Function URL invocation failed:', urlError);
          setError('Lambda invocation failed: ${urlError.message}');
        }
      } else {
        // Try using AWS SDK
        try {
          const lambdaResponse = await lambda.invoke({
            FunctionName: LAMBDA_FUNCTION_NAME,
            InvocationType: 'Event', // Async invocation
            Payload: JSON.stringify(lambdaPayload)
          }).promise();
          console.log('Lambda function invoked successfully via SDK:', lambdaResponse);
          lambdaInvoked = true;
        } catch (lambdaError) {
          console.error('Lambda SDK invocation failed:', lambdaError);
          console.error('Error details:', lambdaError.message, lambdaError.code);
          setError('Lambda invocation failed: ${lambdaError.message}');
        }
      }
      
      if (!lambdaInvoked) {
        console.error('Lambda was not invoked');
        setError('Lambda function was not invoked');
        return; // Don't poll if Lambda wasn't invoked
      }

      // Poll for results
      const files = await pollForFilesByIdentifier(id, 120000, 1500);
      const found = ['backswing', 'midswing', 'followthrough'].filter((k) => !!files[k]).length;
      if (found === 0) {
        setError('No output frames found in storage yet');
        return;
      }
      setAnalysisResults(files);
      if (found < 3) {
        setError('Some frames are still processing');
      }

    } catch (err) {
      console.error('Error details:', err);
      console.error('Error stack:', err.stack);
      const errorMessage = err.message || String(err);
      setError('Failed to upload or analyze video: ${errorMessage}');
      
      if (errorMessage.includes('CORS') || errorMessage.includes('cors')) {
        setError('CORS error');
      } else if (errorMessage.includes('Network') || errorMessage.includes('network')) {
        setError('Network error');
      }
    } finally {
      setIsUploading(false);
    }
  };

  return (
    <ScrollView style={styles.container} contentContainerStyle={styles.contentContainer}>
      <View style={styles.header}>
        <Text style={styles.title}>Video Player</Text>
        <View style={styles.buttonWrapper}>
          <Button title="Upload Forehand" onPress={pickVideo} />
        </View>
      </View>
      {selectedVideo && (
        <View style={styles.videoContainer}>
          <Text style={styles.fileName}>{selectedVideo.name}</Text>
          <View style={[styles.videoWrapper, { width: VIDEO_WIDTH, height: VIDEO_HEIGHT }] }>
            <Video
              ref={videoRef}
              style={styles.video}
              source={{ uri: selectedVideo.uri }}
              useNativeControls={false}
              resizeMode={ResizeMode.COVER}
              isLooping
              shouldPlay={false}
              onPlaybackStatusUpdate={handleVideoStatusUpdate}
            />
            <TouchableOpacity 
              style={styles.playButton} 
              onPress={handlePlayPause}
              activeOpacity={0.7}
            >
              <View style={styles.playIcon}>
                {!isPlaying ? (
                  <View style={styles.triangle} />
                ) : (
                  <View style={styles.pauseIcon}>
                    <View style={styles.pauseBar} />
                    <View style={styles.pauseBar} />
                  </View>
                )}
              </View>
            </TouchableOpacity>
          </View>
          <View style={{ marginTop: 16 }}>
            <Button title={isUploading ? 'Analyzing...' : 'Analyze Forehand'} onPress={uploadForehand} disabled={isUploading} />
            {isUploading && <ActivityIndicator style={{ marginTop: 10 }} size="large" color="#007AFF" />}
          </View>
        </View>
      )}
      {error && (
        <Text style={{ color: 'red', marginTop: 20, textAlign: 'center' }}>{error}</Text>
      )}
      {analysisResults && (
        <View style={styles.resultsContainer}>
          <Text style={styles.resultsTitle}>Key Frames</Text>
          {['backswing', 'midswing', 'followthrough'].map((frame) => (
            analysisResults[frame] ? (
              <View key={frame} style={styles.frameContainer}>
                <Text style={styles.frameTitle}>{frame.charAt(0).toUpperCase() + frame.slice(1)}</Text>
                <Text>File Key: {analysisResults[frame].key}</Text>
                <Image 
                  source={{ 
                    uri: analysisResults[frame].url
                  }} 
                  style={styles.frameImage} 
                  resizeMode="contain"
                  onError={(error) => {
                    console.error('Failed to load image for ${frame}:', error.nativeEvent?.error || error);
                    console.error('URL was: ${analysisResults[frame].url}');
                  }}
                  onLoad={() => {
                    console.log('Successfully loaded image for ${frame}');
                  }}
                />
              </View>
            ) : null
          ))}
        </View>
      )}
      <StatusBar style="auto" />
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#fff',
  },
  contentContainer: {
    flexGrow: 1,
    alignItems: 'center',
    padding: 16,
  },
  header: {
    marginTop: 40,
    marginBottom: 24,
    width: '100%',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  buttonWrapper: {
    marginBottom: 8,
  },
  videoContainer: {
    marginTop: 10,
    width: '100%',
    alignItems: 'center',
  },
  videoWrapper: {
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    overflow: 'hidden',
    position: 'relative',
    backgroundColor: '#000',
    justifyContent: 'center',
    alignItems: 'center',
  },
  video: {
    width: '100%',
    height: '100%',
    position: 'absolute',
    top: 0,
    left: 0,
  },
  playButton: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
  },
  playIcon: {
    width: 60,
    height: 60,
    backgroundColor: 'rgba(0, 0, 0, 0.7)',
    borderRadius: 30,
    justifyContent: 'center',
    alignItems: 'center',
  },
  triangle: {
    width: 0,
    height: 0,
    backgroundColor: 'transparent',
    borderStyle: 'solid',
    borderLeftWidth: 15,
    borderRightWidth: 0,
    borderBottomWidth: 10,
    borderTopWidth: 10,
    borderLeftColor: 'white',
    borderRightColor: 'transparent',
    borderBottomColor: 'transparent',
    borderTopColor: 'transparent',
    marginLeft: 3,
  },
  pauseIcon: {
    flexDirection: 'row',
    justifyContent: 'center',
    alignItems: 'center',
    width: 20,
    height: 20,
  },
  pauseBar: {
    width: 3,
    height: 20,
    backgroundColor: 'white',
    marginHorizontal: 1,
  },
  fileName: {
    textAlign: 'center',
    marginBottom: 10,
    fontSize: 16,
    color: '#333',
    fontWeight: '500',
  },
  resultsContainer: {
    marginTop: 30,
    width: '100%',
    alignItems: 'center',
  },
  resultsTitle: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 16,
    textAlign: 'center',
  },
  frameContainer: {
    marginBottom: 20,
    alignItems: 'center',
  },
  frameTitle: {
    fontSize: 18,
    fontWeight: 'bold',
    marginBottom: 10,
  },
  frameImage: {
    width: VIDEO_WIDTH,
    height: 200,
    borderWidth: 1,
    borderColor: '#ccc',
    borderRadius: 8,
    backgroundColor: '#000',
  },
});
