// GENERATED CODE - DO NOT MODIFY BY HAND

part of 'cooperative_membership_model.dart';

// **************************************************************************
// TypeAdapterGenerator
// **************************************************************************

class CooperativeMembershipAdapter extends TypeAdapter<CooperativeMembership> {
  @override
  final int typeId = 1;

  @override
  CooperativeMembership read(BinaryReader reader) {
    final numOfFields = reader.readByte();
    final fields = <int, dynamic>{
      for (int i = 0; i < numOfFields; i++) reader.readByte(): reader.read(),
    };
    return CooperativeMembership(
      cooperativeId: fields[0] as String,
      role: fields[1] as String,
    );
  }

  @override
  void write(BinaryWriter writer, CooperativeMembership obj) {
    writer
      ..writeByte(2)
      ..writeByte(0)
      ..write(obj.cooperativeId)
      ..writeByte(1)
      ..write(obj.role);
  }

  @override
  int get hashCode => typeId.hashCode;

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is CooperativeMembershipAdapter &&
          runtimeType == other.runtimeType &&
          typeId == other.typeId;
}
