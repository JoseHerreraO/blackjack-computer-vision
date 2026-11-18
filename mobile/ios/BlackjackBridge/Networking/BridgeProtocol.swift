//
// BridgeProtocol.swift
//
// Wire format between the iPhone (TCP client) and the PC (TCP server, port 5050).
// This is a contract with the PC side: do not change it without telling the team.
//
// Every message is [1 byte type][uint32 big-endian payload length][payload].
//
//   iPhone -> PC
//     0x01 HELLO   JSON {"device","width","height","fps","jpeg_quality","app_version"}.
//                  Sent right after connecting and whenever the resolution or fps changes.
//     0x02 FRAME   int64 big-endian capture time (ms since the Unix epoch) + JPEG bytes.
//                  The JPEG is the frame as the glasses deliver it: not rotated, cropped
//                  or scaled.
//     0x03 LOG     UTF-8 text, one log line.
//
//   PC -> iPhone
//     0x10 COMMAND JSON {"type":"say","text":"...","lang":"es-ES"} or
//                  {"type":"status","text":"..."}.
//

import Foundation

enum MessageType: UInt8, Sendable {
    case hello = 0x01
    case frame = 0x02
    case log = 0x03
    case command = 0x10
}

enum BridgeProtocol {
    static let headerLength = 5
    /// Upper bound for messages coming from the PC (commands are tiny).
    static let maxIncomingPayload = 1 << 20
    static let jpegQuality = 0.8

    struct Hello: Encodable, Equatable, Sendable {
        let device: String
        let width: Int
        let height: Int
        let fps: Int
        let jpegQuality: Double
        let appVersion: String

        enum CodingKeys: String, CodingKey {
            case device, width, height, fps
            case jpegQuality = "jpeg_quality"
            case appVersion = "app_version"
        }
    }

    struct Command: Decodable, Sendable {
        let type: String
        let text: String?
        let lang: String?
    }

    static func message(_ type: MessageType, payload: Data) -> Data {
        var data = Data(capacity: headerLength + payload.count)
        data.append(type.rawValue)
        appendBigEndian(UInt32(payload.count), to: &data)
        data.append(payload)
        return data
    }

    static func frame(capturedAtMs: Int64, jpeg: Data) -> Data {
        var payload = Data(capacity: 8 + jpeg.count)
        appendBigEndian(capturedAtMs, to: &payload)
        payload.append(jpeg)
        return message(.frame, payload: payload)
    }

    /// Decodes the 5-byte header into the raw type byte and the payload length.
    static func parseHeader(_ data: Data) -> (type: UInt8, length: Int)? {
        guard data.count == headerLength else { return nil }
        let bytes = [UInt8](data)
        let length = bytes[1...4].reduce(0) { $0 << 8 | Int($1) }
        return (bytes[0], length)
    }

    /// Hardware model and iOS version, e.g. "iPhone16,1 (iOS 26.1.0)".
    static let deviceDescription: String = {
        var info = utsname()
        uname(&info)
        let machine = withUnsafeBytes(of: &info.machine) { buffer in
            String(decoding: buffer.prefix { $0 != 0 }, as: UTF8.self)
        }
        let os = ProcessInfo.processInfo.operatingSystemVersion
        return "\(machine) (iOS \(os.majorVersion).\(os.minorVersion).\(os.patchVersion))"
    }()

    private static func appendBigEndian<T: FixedWidthInteger>(_ value: T, to data: inout Data) {
        withUnsafeBytes(of: value.bigEndian) { data.append(contentsOf: $0) }
    }
}
