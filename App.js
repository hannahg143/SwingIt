import { StatusBar } from 'expo-status-bar';
import { StyleSheet, Text, View, Button, Dimensions, Alert, ScrollView, TouchableOpacity, Image, ActivityIndicator, Platform } from 'react-native';
import { useState, useRef } from 'react';
import { Video, ResizeMode } from 'expo-av';
import * as DocumentPicker from 'expo-document-picker';
import { Client, Storage, Functions, Account } from 'appwrite';

const APPWRITE_ENDPOINT = 'https://fra.cloud.appwrite.io/v1';
const APPWRITE_PROJECT_ID = '683e23ab0003b764a043';
const APPWRITE_BUCKET_ID = '685a225b00147a6dd7b2';
const APPWRITE_FUNCTION_ID = '685a232a0029d6579690';
const APPWRITE_API_KEY = 'standard_6e82b0c7de2010567c8da026ded3827a1b2e68137565469a5aea8ee7f63fc0ac3d034aad0a46e67d308fe479dd97047059b97c81264627f8b7f8d3b23d4f79620b2d3adda03402b1c3174b962c2545b7ef1c52457fe63a66633d86f740f71478f3422e74a5b61acbc355e7888standard_6e82b0c7de2010567c8da026ded3827a1b2e68137565469a5aea8ee7f63fc0ac3d034aad0a46e67d308fe479dd97047059b97c81264627f8b7f8d3b23d4f79620b2d3adda03402b1c3174b962c2545b7ef1c52457fe63a66633d86f740f71478f3422e74a5b61acbc355e7888a3dd28ebed0d18850f9301991351ad2923c847aa3dd28ebed0d18850f9301991351ad2923c847a';

const windowWidth = Dimensions.get('window').width;
const windowHeight = Dimensions.get('window').height;

const isMobile = windowWidth < 600;
const VIDEO_WIDTH = isMobile ? windowWidth - 32 : Math.min(windowWidth * 0.95, 900);
const VIDEO_HEIGHT = isMobile ? Math.min((windowWidth - 32) * 16 / 9, windowHeight * 0.5) : 500;

const client = new Client()
  .setEndpoint(APPWRITE_ENDPOINT)
  .setProject(APPWRITE_PROJECT_ID);


const storage = new Storage(client);
const functions = new Functions(client);
const account = new Account(client);


const createSession = async () => {
  try {
    await account.createAnonymousSession();
  } catch (error) {
  }
};

createSession();

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
      console.error('Error picking video:', error);
    }
  };

  const wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

  const pollForFilesByIdentifier = async (id, timeoutMs = 120000, intervalMs = 1500) => {
    const expected = {
      backswing: `${id}1`,
      midswing: `${id}2`,
      followthrough: `${id}3`,
    };
    const files = { backswing: null, midswing: null, followthrough: null };
    const deadline = Date.now() + timeoutMs;


    while (Date.now() < deadline) {
      for (const key of Object.keys(expected)) {
        if (!files[key]) {
          try {
            const file = await storage.getFile(APPWRITE_BUCKET_ID, expected[key]);
            files[key] = file;
          } catch (error) {
          }
        }
      }
      if (files.backswing && files.midswing && files.followthrough) break;
      await wait(intervalMs);
    }

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

  const getAppwriteFile = async (video) => {
    if (Platform.OS === 'web') {
      const response = await fetch(video.uri);
      const blob = await response.blob();
      return new File([blob], video.name || 'video.mp4', { type: blob.type || 'video/mp4' });
    } else {
      return {
        uri: video.uri,
        name: video.name || 'video.mp4',
        type: 'video/mp4',
      };
    }
  };

  const uploadForehand = async () => {
    if (!selectedVideo) {
      setError('Please select a video first.');
      return;
    }

    setIsUploading(true);
    setError('');

    try {
      const appwriteFile = await getAppwriteFile(selectedVideo);
      const uploadResponse = await storage.createFile(
        APPWRITE_BUCKET_ID,
        'unique()',
        appwriteFile
      );

      const id = String(Date.now());
      setIdentifier(id);
      const execution = await functions.createExecution(
        APPWRITE_FUNCTION_ID,
        JSON.stringify({ 
          fileId: uploadResponse.$id, 
          userId: 'user', 
          backswingId: `${id}1`,
          midswingId: `${id}2`,
          followthroughId: `${id}3`
        }),
        true
      );

      const files = await pollForFilesByIdentifier(id, 120000, 1500);
      const found = ['backswing', 'midswing', 'followthrough'].filter((k) => !!files[k]).length;
      if (found === 0) {
        setError('No output frames found in storage yet. Please try again shortly.');
        return;
      }
      setAnalysisResults(files);
      if (found < 3) {
        setError('Some frames are still processing. Missing frames will appear once ready.');
      }

    } catch (err) {
      console.error('Error:', err);
      setError('Failed to upload or analyze video.');
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
                <Text>File ID: {analysisResults[frame].$id}</Text>
                <Text>File Name: {analysisResults[frame].name}</Text>
                <Image 
                  source={{ 
                    uri: `${APPWRITE_ENDPOINT}/storage/buckets/${APPWRITE_BUCKET_ID}/files/${analysisResults[frame].$id}/view?project=${APPWRITE_PROJECT_ID}` 
                  }} 
                  style={styles.frameImage} 
                  resizeMode="contain" 
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
