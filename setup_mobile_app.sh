#!/bin/bash
# Mobile Companion App Setup
# This scaffold provides a foundation for building native iOS/Android or React Native apps

# Create mobile app directory structure
mkdir -p mobile_app/{ios,android,react_native,api_client}

cat > mobile_app/API_SPEC.md << 'EOF'
# Smart Traffic Mobile Companion App - API Specification

## Overview
Mobile officers use a companion app to review/action violations in the field, with offline support and photo integration.

## Core Endpoints

### Authentication
- POST `/api/mobile/login` - Login with credentials
- POST `/api/mobile/logout` - Logout
- POST `/api/mobile/refresh_token` - Refresh auth token

### Violations Review (Officer)
- GET `/api/mobile/violations?status=new` - Get violations to review
- POST `/api/mobile/violations/{id}/action` - Take action on violation
  - Body: `{action: 'approved'|'rejected'|'send', notes: '...'}`
- GET `/api/mobile/violations/{id}/evidence` - Download evidence image
- POST `/api/mobile/violations/{id}/photo_capture` - Add officer photo

### Offline Sync
- GET `/api/mobile/sync/pull` - Get delta of new violations since last sync
- POST `/api/mobile/sync/push` - Upload offline actions to cloud
- GET `/api/mobile/sync/status` - Get sync status

### Location & Navigation
- GET `/api/mobile/hotspots` - Get high-violation intersections nearby
- GET `/api/mobile/assignments` - Get officer's assigned routes

### Profile & Settings
- GET `/api/mobile/user/profile` - Get current user profile
- POST `/api/mobile/user/preferences` - Save preferences (theme, notifications)

## Response Format
```json
{
  "ok": true,
  "data": {...},
  "error": null,
  "timestamp": "2026-03-24T10:00:00Z"
}
```

## Mobile-First Considerations
1. All responses keep payload < 100KB for mobile data efficiency
2. Images served in mobile-optimized formats (WebP, compressed JPEG)
3. Pagination with cursor-based navigation
4. Offline-first: all data stored in SQLite on device
5. Sync via background service when WiFi available
EOF

cat > mobile_app/IMPLEMENTATION_NOTES.md << 'EOF'
# Implementation Guide

## Recommended Tech Stack

### Option 1: React Native (Cross-platform)
- Framework: React Native + Expo
- State: Redux or Zustand
- Offline: WatermelonDB (SQLite wrapper)
- Maps: react-native-maps
- Estimated effort: 4-6 weeks

### Option 2: Native (Better performance)
- iOS: Swift + SwiftUI
- Android: Kotlin + Jetpack Compose
- Offline: Realm DB
- Maps: MapKit (iOS), Google Maps (Android)
- Estimated effort: 8-12 weeks

### Option 3: Flutter (Single codebase, native performance)
- Framework: Flutter + Dart
- State: Provider or Riverpod
- Offline: Isar DB
- Maps: google_maps_flutter
- Estimated effort: 5-7 weeks

## Core Features to Implement
1. Police Authentication (login/session mgmt)
2. Violation List View with real-time updates
3. Evidence Image Carousel Viewer
4. Status Update Workflow (New → Reviewed → Sent)
5. Photo Capture & Annotation (add officer photo/notes)
6. Offline Sync Queue
7. Geolocation-based Hotspots
8. Settings & Preferences
9. Push Notifications
10. Analytics Dashboard

## Backend API Compatibility
- All mobile endpoints should be versioned: `/api/mobile/v1/...`
- Use JWT tokens with 24h expiry for mobile auth
- CORS enabled for mobile origins
- Rate limiting: 100 req/min per officer

## Data Security
- Encrypt SQLite on device (SQLCipher for Android, Keychain for iOS)
- Use TLS for all API calls
- Implement certificate pinning
- Store credentials in secure enclave (iOS) / Keystore (Android)
- Implement auto-logout after 15 mins inactivity

## Testing
- Unit tests: Jest (React Native) or XCTest (iOS) / Espresso (Android)
- Integration: Detox (React Native)
- E2E: Manual testing on physical devices
EOF

echo "✓ Mobile app scaffold created in mobile_app/"
echo ""
echo "API Specification: mobile_app/API_SPEC.md"
echo "Implementation Notes: mobile_app/IMPLEMENTATION_NOTES.md"
echo ""
echo "Next: Choose tech stack and initialize project"
EOF
