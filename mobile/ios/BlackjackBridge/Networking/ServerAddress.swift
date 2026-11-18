//
// ServerAddress.swift
//
// Address of the PC receiver as typed in the app: "host:port" or just "host".
//

import Foundation

struct ServerAddress: Equatable, Sendable, CustomStringConvertible {
    /// Port of the PC receiver (TCP server). Part of the iPhone <-> PC contract.
    static let defaultPort: UInt16 = 5050

    let host: String
    let port: UInt16

    /// Parses "192.168.1.50:5050" or "192.168.1.50" (default port). IPv6 is not supported.
    init?(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let parts = trimmed.split(separator: ":", omittingEmptySubsequences: false)
        guard let first = parts.first, !first.isEmpty, !first.contains(" "), parts.count <= 2 else {
            return nil
        }
        if parts.count == 2 {
            guard let port = UInt16(parts[1]), port > 0 else { return nil }
            self.port = port
        } else {
            self.port = Self.defaultPort
        }
        self.host = String(first)
    }

    var description: String { "\(host):\(port)" }
}
