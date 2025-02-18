import 'package:cloud_firestore/cloud_firestore.dart';

enum NotificationType {
  cooperativeInvite,
  generalMessage,
}

enum NotificationAction {
  acceptInvite,
  declineInvite,
  none,
}

class NotificationModel {
  final String id;
  final String userId;
  final NotificationType type;
  final String title;
  final String message;
  final String? cooperativeId;
  final DateTime createdAt;
  final bool isRead;
  final NotificationAction action;

  NotificationModel({
    required this.id,
    required this.userId,
    required this.type,
    required this.title,
    required this.message,
    this.cooperativeId,
    required this.createdAt,
    required this.isRead,
    required this.action,
  });

  Map<String, dynamic> toMap() {
    return {
      'userId': userId,
      'type': type.name,
      'title': title,
      'message': message,
      'cooperativeId': cooperativeId,
      'createdAt': Timestamp.fromDate(createdAt),
      'isRead': isRead,
      'action': action.name,
    };
  }

  factory NotificationModel.fromMap(String id, Map<String, dynamic> map) {
    return NotificationModel(
      id: id,
      userId: map['userId'] ?? '',
      type: NotificationType.values.firstWhere(
        (e) => e.name == map['type'],
        orElse: () => NotificationType.generalMessage,
      ),
      title: map['title'] ?? '',
      message: map['message'] ?? '',
      cooperativeId: map['cooperativeId'],
      createdAt: (map['createdAt'] as Timestamp).toDate(),
      isRead: map['isRead'] ?? false,
      action: NotificationAction.values.firstWhere(
        (e) => e.name == map['action'],
        orElse: () => NotificationAction.none,
      ),
    );
  }

  factory NotificationModel.cooperativeInvite({
    required String id,
    required String userId,
    required String cooperativeName,
    required String cooperativeId,
    required String invitedBy,
  }) {
    return NotificationModel(
      id: id,
      userId: userId,
      type: NotificationType.cooperativeInvite,
      title: 'Cooperative Invitation',
      message: '$invitedBy invited you to join $cooperativeName',
      cooperativeId: cooperativeId,
      createdAt: DateTime.now(),
      isRead: false,
      action: NotificationAction.acceptInvite,
    );
  }

  factory NotificationModel.generalMessage({
    required String id,
    required String userId,
    required String title,
    required String message,
  }) {
    return NotificationModel(
      id: id,
      userId: userId,
      type: NotificationType.generalMessage,
      title: title,
      message: message,
      createdAt: DateTime.now(),
      isRead: false,
      action: NotificationAction.none,
    );
  }

  NotificationModel copyWith({
    String? id,
    String? userId,
    NotificationType? type,
    String? title,
    String? message,
    String? cooperativeId,
    DateTime? createdAt,
    bool? isRead,
    NotificationAction? action,
  }) {
    return NotificationModel(
      id: id ?? this.id,
      userId: userId ?? this.userId,
      type: type ?? this.type,
      title: title ?? this.title,
      message: message ?? this.message,
      cooperativeId: cooperativeId ?? this.cooperativeId,
      createdAt: createdAt ?? this.createdAt,
      isRead: isRead ?? this.isRead,
      action: action ?? this.action,
    );
  }
}
