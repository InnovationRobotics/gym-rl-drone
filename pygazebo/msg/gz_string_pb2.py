# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: gz_string.proto

import sys
_b=sys.version_info[0]<3 and (lambda x:x) or (lambda x:x.encode('latin1'))
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
from google.protobuf import descriptor_pb2
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor.FileDescriptor(
  name='gz_string.proto',
  package='gazebo.msgs',
  serialized_pb=_b('\n\x0fgz_string.proto\x12\x0bgazebo.msgs\"\x18\n\x08GzString\x12\x0c\n\x04\x64\x61ta\x18\x01 \x02(\t')
)
_sym_db.RegisterFileDescriptor(DESCRIPTOR)




_GZSTRING = _descriptor.Descriptor(
  name='GzString',
  full_name='gazebo.msgs.GzString',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  fields=[
    _descriptor.FieldDescriptor(
      name='data', full_name='gazebo.msgs.GzString.data', index=0,
      number=1, type=9, cpp_type=9, label=2,
      has_default_value=False, default_value=_b("").decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      options=None),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  options=None,
  is_extendable=False,
  extension_ranges=[],
  oneofs=[
  ],
  serialized_start=32,
  serialized_end=56,
)

DESCRIPTOR.message_types_by_name['GzString'] = _GZSTRING

GzString = _reflection.GeneratedProtocolMessageType('GzString', (_message.Message,), dict(
  DESCRIPTOR = _GZSTRING,
  __module__ = 'gz_string_pb2'
  # @@protoc_insertion_point(class_scope:gazebo.msgs.GzString)
  ))
_sym_db.RegisterMessage(GzString)


# @@protoc_insertion_point(module_scope)
