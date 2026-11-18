//
// StreamSettings.swift
//
// Resolutions and frame rates the glasses can stream (Meta Wearables DAT 1.0.1).
// Measured on 2026-11-17 with Ray-Ban Meta: 720x1280 gives a steady 15 fps, and asking
// for 24 fps at that resolution still delivers 15 fps, so 720x1280 @ 15 fps is the default.
//

import MWDATCamera

enum StreamResolution: String, CaseIterable, Identifiable, Sendable {
    case high
    case medium
    case low

    var id: String { rawValue }

    /// Frame size as delivered by DAT: portrait, not rotated.
    var width: Int {
        switch self {
        case .high: 720
        case .medium: 504
        case .low: 360
        }
    }

    var height: Int {
        switch self {
        case .high: 1280
        case .medium: 896
        case .low: 640
        }
    }

    var label: String { "\(width)×\(height)" }

    var datResolution: StreamingResolution {
        switch self {
        case .high: .high
        case .medium: .medium
        case .low: .low
        }
    }
}

enum StreamSettings {
    /// Frame rates accepted by DAT.
    static let frameRates: [UInt] = [2, 7, 15, 24, 30]
    static let defaultResolution = StreamResolution.high
    static let defaultFrameRate: UInt = 15

    /// Raw (decoded) frames, so each one can be encoded to JPEG for the PC.
    static func configuration(resolution: StreamResolution, frameRate: UInt) -> StreamConfiguration {
        StreamConfiguration(
            videoCodec: .raw,
            resolution: resolution.datResolution,
            frameRate: frameRate
        )
    }
}
