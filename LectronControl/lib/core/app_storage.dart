import 'dart:convert';
import 'dart:io';

import 'models.dart';

/// Everything the app keeps on disk lives in its own data directory: the daily log files and the
/// remembered settings. Nothing is created in the user's home directory.
class AppStorage {
  AppStorage(this.root);

  /// Null when no directory is available; the app then runs without persistence.
  final Directory? root;

  IOSink? _sink;
  String? _sinkDay;
  Map<String, Object?> _settings = {};

  Directory? get logDir => root == null ? null : Directory('${root!.path}${Platform.pathSeparator}logs');

  String? get logPath {
    final dir = logDir;
    if (dir == null) return null;
    final now = DateTime.now();
    return '${dir.path}${Platform.pathSeparator}board_test_${now.year}${two(now.month)}${two(now.day)}.log';
  }

  File? get _settingsFile => root == null ? null : File('${root!.path}${Platform.pathSeparator}settings.json');

  void load() {
    try {
      final file = _settingsFile;
      if (file != null && file.existsSync()) {
        _settings = Map<String, Object?>.from(jsonDecode(file.readAsStringSync()) as Map);
      }
    } catch (_) {
      _settings = {};
    }
  }

  String? getString(String key) => _settings[key] as String?;
  int? getInt(String key) => _settings[key] as int?;

  void put(String key, Object? value) {
    if (_settings[key] == value) return;
    _settings[key] = value;
    try {
      _settingsFile?.writeAsStringSync(const JsonEncoder.withIndent('  ').convert(_settings));
    } catch (_) {
      // settings are a convenience, never an error
    }
  }

  /// Appends one line to today's log file.
  void log(String level, String message) {
    final path = logPath;
    if (path == null) return;
    try {
      if (_sink == null || _sinkDay != path) {
        _sink?.close();
        logDir!.createSync(recursive: true);
        _sink = File(path).openWrite(mode: FileMode.append);
        _sinkDay = path;
      }
      final now = DateTime.now();
      _sink!.writeln('${ymd(now)} ${hms(now)} ${level.padRight(7)} $message');
    } catch (_) {
      _sink = null;
    }
  }

  Future<void> close() async {
    final sink = _sink;
    _sink = null;
    await sink?.flush();
    await sink?.close();
  }
}
