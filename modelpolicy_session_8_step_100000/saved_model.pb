ƒЏ
ѕ≥
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
≥
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetypeИ
Њ
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring И
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshapeИ"serve*2.1.02unknown8пы
Ж
QNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:d*&
shared_nameQNetwork/dense/kernel

)QNetwork/dense/kernel/Read/ReadVariableOpReadVariableOpQNetwork/dense/kernel*
_output_shapes

:d*
dtype0
~
QNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameQNetwork/dense/bias
w
'QNetwork/dense/bias/Read/ReadVariableOpReadVariableOpQNetwork/dense/bias*
_output_shapes
:*
dtype0
«
5QNetwork/EncodingNetwork/EncodingNetwork/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	Дd*F
shared_name75QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel
ј
IQNetwork/EncodingNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpReadVariableOp5QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel*
_output_shapes
:	Дd*
dtype0
Њ
3QNetwork/EncodingNetwork/EncodingNetwork/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:d*D
shared_name53QNetwork/EncodingNetwork/EncodingNetwork/dense/bias
Ј
GQNetwork/EncodingNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpReadVariableOp3QNetwork/EncodingNetwork/EncodingNetwork/dense/bias*
_output_shapes
:d*
dtype0
P
ConstConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€

NoOpNoOp
љ
Const_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*ц
valueмBй Bв
%
_wrapped_policy

signatures


_q_network
 
t
_encoder
_q_value_layer
	variables
trainable_variables
regularization_losses
		keras_api
n

_postprocessing_layers
	variables
trainable_variables
regularization_losses
	keras_api
h

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api

0
1
2
3

0
1
2
3
 
Ъ
	variables
non_trainable_variables
layer_regularization_losses
trainable_variables

layers
metrics
regularization_losses

0
1

0
1

0
1
 
Ъ
	variables
non_trainable_variables
layer_regularization_losses
trainable_variables

layers
 metrics
regularization_losses
vt
VARIABLE_VALUEQNetwork/dense/kernelK_wrapped_policy/_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUEQNetwork/dense/biasI_wrapped_policy/_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
Ъ
	variables
!non_trainable_variables
"layer_regularization_losses
trainable_variables

#layers
$metrics
regularization_losses
НК
VARIABLE_VALUE5QNetwork/EncodingNetwork/EncodingNetwork/dense/kernelA_wrapped_policy/_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUE
ЛИ
VARIABLE_VALUE3QNetwork/EncodingNetwork/EncodingNetwork/dense/biasA_wrapped_policy/_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUE
 
 

0
1
 
R
%	variables
&trainable_variables
'regularization_losses
(	keras_api
h

kernel
bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
 
 

0
1
 
 
 
 
 
 
 
 
Ъ
%	variables
-non_trainable_variables
.layer_regularization_losses
&trainable_variables

/layers
0metrics
'regularization_losses

0
1

0
1
 
Ъ
)	variables
1non_trainable_variables
2layer_regularization_losses
*trainable_variables

3layers
4metrics
+regularization_losses
 
 
 
 
 
 
 
 
l
action_0/discountPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€

action_0/observationPlaceholder*+
_output_shapes
:€€€€€€€€€
*
dtype0* 
shape:€€€€€€€€€

j
action_0/rewardPlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
m
action_0/step_typePlaceholder*#
_output_shapes
:€€€€€€€€€*
dtype0*
shape:€€€€€€€€€
у
StatefulPartitionedCallStatefulPartitionedCallaction_0/discountaction_0/observationaction_0/rewardaction_0/step_type5QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel3QNetwork/EncodingNetwork/EncodingNetwork/dense/biasQNetwork/dense/kernelQNetwork/dense/bias*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*/
f*R(
&__inference_signature_wrapper_30108382
]
get_initial_state_batch_sizePlaceholder*
_output_shapes
: *
dtype0*
shape: 
№
PartitionedFunctionCallPartitionedCallget_initial_state_batch_size*
Tin
2*

Tout
 *,
_gradient_op_typePartitionedCallUnused*
config_proto	RRШ*/
f*R(
&__inference_signature_wrapper_30108393
м
PartitionedCallPartitionedCallConst*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*/
f*R(
&__inference_signature_wrapper_30108405
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
ш
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename)QNetwork/dense/kernel/Read/ReadVariableOp'QNetwork/dense/bias/Read/ReadVariableOpIQNetwork/EncodingNetwork/EncodingNetwork/dense/kernel/Read/ReadVariableOpGQNetwork/EncodingNetwork/EncodingNetwork/dense/bias/Read/ReadVariableOpConst_1*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8**
f%R#
!__inference__traced_save_30108460
°
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameQNetwork/dense/kernelQNetwork/dense/bias5QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel3QNetwork/EncodingNetwork/EncodingNetwork/dense/bias*
Tin	
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*-
f(R&
$__inference__traced_restore_30108484№Ћ
ЩN
„
__inference_action_264
	time_step
time_step_1
time_step_2
time_step_3Q
Mqnetwork_encodingnetwork_encodingnetwork_dense_matmul_readvariableop_resourceR
Nqnetwork_encodingnetwork_encodingnetwork_dense_biasadd_readvariableop_resource1
-qnetwork_dense_matmul_readvariableop_resource2
.qnetwork_dense_biasadd_readvariableop_resource
identityИҐEQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpҐDQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpҐ%QNetwork/dense/BiasAdd/ReadVariableOpҐ$QNetwork/dense/MatMul/ReadVariableOp°
&QNetwork/EncodingNetwork/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"€€€€  2(
&QNetwork/EncodingNetwork/flatten/Const–
(QNetwork/EncodingNetwork/flatten/ReshapeReshapetime_step_3/QNetwork/EncodingNetwork/flatten/Const:output:0*
T0*(
_output_shapes
:€€€€€€€€€Д2*
(QNetwork/EncodingNetwork/flatten/Reshapeз
3QNetwork/EncodingNetwork/EncodingNetwork/dense/CastCast1QNetwork/EncodingNetwork/flatten/Reshape:output:0*

DstT0*

SrcT0*(
_output_shapes
:€€€€€€€€€Д25
3QNetwork/EncodingNetwork/EncodingNetwork/dense/CastЫ
DQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpReadVariableOpMqnetwork_encodingnetwork_encodingnetwork_dense_matmul_readvariableop_resource*
_output_shapes
:	Дd*
dtype02F
DQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp±
5QNetwork/EncodingNetwork/EncodingNetwork/dense/MatMulMatMul7QNetwork/EncodingNetwork/EncodingNetwork/dense/Cast:y:0LQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d27
5QNetwork/EncodingNetwork/EncodingNetwork/dense/MatMulЩ
EQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpReadVariableOpNqnetwork_encodingnetwork_encodingnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:d*
dtype02G
EQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpљ
6QNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAddBiasAdd?QNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul:product:0MQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€d28
6QNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAddе
3QNetwork/EncodingNetwork/EncodingNetwork/dense/ReluRelu?QNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd:output:0*
T0*'
_output_shapes
:€€€€€€€€€d25
3QNetwork/EncodingNetwork/EncodingNetwork/dense/ReluЇ
$QNetwork/dense/MatMul/ReadVariableOpReadVariableOp-qnetwork_dense_matmul_readvariableop_resource*
_output_shapes

:d*
dtype02&
$QNetwork/dense/MatMul/ReadVariableOpџ
QNetwork/dense/MatMulMatMulAQNetwork/EncodingNetwork/EncodingNetwork/dense/Relu:activations:0,QNetwork/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
QNetwork/dense/MatMulє
%QNetwork/dense/BiasAdd/ReadVariableOpReadVariableOp.qnetwork_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02'
%QNetwork/dense/BiasAdd/ReadVariableOpљ
QNetwork/dense/BiasAddBiasAddQNetwork/dense/MatMul:product:0-QNetwork/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:€€€€€€€€€2
QNetwork/dense/BiasAddk
ExpandDims/dimConst*
_output_shapes
: *
dtype0*
valueB :
ю€€€€€€€€2
ExpandDims/dimЦ

ExpandDims
ExpandDimsQNetwork/dense/BiasAdd:output:0ExpandDims/dim:output:0*
T0*+
_output_shapes
:€€€€€€€€€2

ExpandDims£
*ShiftedCategorical_1/mode/ArgMax/dimensionConst*
_output_shapes
: *
dtype0*
valueB :
€€€€€€€€€2,
*ShiftedCategorical_1/mode/ArgMax/dimension 
 ShiftedCategorical_1/mode/ArgMaxArgMaxExpandDims:output:03ShiftedCategorical_1/mode/ArgMax/dimension:output:0*
T0*'
_output_shapes
:€€€€€€€€€2"
 ShiftedCategorical_1/mode/ArgMaxі
ShiftedCategorical_1/mode/CastCast)ShiftedCategorical_1/mode/ArgMax:output:0*

DstT0*

SrcT0	*'
_output_shapes
:€€€€€€€€€2 
ShiftedCategorical_1/mode/CastP
add/yConst*
_output_shapes
: *
dtype0*
value	B : 2
add/yy
addAddV2"ShiftedCategorical_1/mode/Cast:y:0add/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
addj
Deterministic/atolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/atolj
Deterministic/rtolConst*
_output_shapes
: *
dtype0*
value	B : 2
Deterministic/rtolС
%Deterministic_1/sample/sample_shape/xConst*
_output_shapes
: *
dtype0*
valueB 2'
%Deterministic_1/sample/sample_shape/xі
#Deterministic_1/sample/sample_shapeCast.Deterministic_1/sample/sample_shape/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 2%
#Deterministic_1/sample/sample_shapes
Deterministic_1/sample/ShapeShapeadd:z:0*
T0*
_output_shapes
:2
Deterministic_1/sample/ShapeГ
Deterministic_1/sample/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_1Г
Deterministic_1/sample/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 2 
Deterministic_1/sample/Shape_2…
$Deterministic_1/sample/BroadcastArgsBroadcastArgs'Deterministic_1/sample/Shape_1:output:0'Deterministic_1/sample/Shape_2:output:0*
_output_shapes
: 2&
$Deterministic_1/sample/BroadcastArgsѕ
&Deterministic_1/sample/BroadcastArgs_1BroadcastArgs%Deterministic_1/sample/Shape:output:0)Deterministic_1/sample/BroadcastArgs:r0:0*
_output_shapes
:2(
&Deterministic_1/sample/BroadcastArgs_1
Deterministic_1/sample/ConstConst*
_output_shapes
: *
dtype0*
valueB 2
Deterministic_1/sample/ConstЪ
&Deterministic_1/sample/concat/values_0Const*
_output_shapes
:*
dtype0*
valueB:2(
&Deterministic_1/sample/concat/values_0К
"Deterministic_1/sample/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : 2$
"Deterministic_1/sample/concat/axis™
Deterministic_1/sample/concatConcatV2/Deterministic_1/sample/concat/values_0:output:0+Deterministic_1/sample/BroadcastArgs_1:r0:0%Deterministic_1/sample/Const:output:0+Deterministic_1/sample/concat/axis:output:0*
N*
T0*
_output_shapes
:2
Deterministic_1/sample/concat«
"Deterministic_1/sample/BroadcastToBroadcastToadd:z:0&Deterministic_1/sample/concat:output:0*
T0*4
_output_shapes"
 :€€€€€€€€€€€€€€€€€€2$
"Deterministic_1/sample/BroadcastToЫ
Deterministic_1/sample/Shape_3Shape+Deterministic_1/sample/BroadcastTo:output:0*
T0*
_output_shapes
:2 
Deterministic_1/sample/Shape_3Ґ
*Deterministic_1/sample/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2,
*Deterministic_1/sample/strided_slice/stack¶
,Deterministic_1/sample/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB: 2.
,Deterministic_1/sample/strided_slice/stack_1¶
,Deterministic_1/sample/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,Deterministic_1/sample/strided_slice/stack_2к
$Deterministic_1/sample/strided_sliceStridedSlice'Deterministic_1/sample/Shape_3:output:03Deterministic_1/sample/strided_slice/stack:output:05Deterministic_1/sample/strided_slice/stack_1:output:05Deterministic_1/sample/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:*
end_mask2&
$Deterministic_1/sample/strided_sliceО
$Deterministic_1/sample/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 2&
$Deterministic_1/sample/concat_1/axisГ
Deterministic_1/sample/concat_1ConcatV2'Deterministic_1/sample/sample_shape:y:0-Deterministic_1/sample/strided_slice:output:0-Deterministic_1/sample/concat_1/axis:output:0*
N*
T0*
_output_shapes
:2!
Deterministic_1/sample/concat_1‘
Deterministic_1/sample/ReshapeReshape+Deterministic_1/sample/BroadcastTo:output:0(Deterministic_1/sample/concat_1:output:0*
T0*'
_output_shapes
:€€€€€€€€€2 
Deterministic_1/sample/Reshapet
clip_by_value/Minimum/yConst*
_output_shapes
: *
dtype0*
value	B :2
clip_by_value/Minimum/yґ
clip_by_value/MinimumMinimum'Deterministic_1/sample/Reshape:output:0 clip_by_value/Minimum/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
clip_by_value/Minimumd
clip_by_value/yConst*
_output_shapes
: *
dtype0*
value	B : 2
clip_by_value/yР
clip_by_valueMaximumclip_by_value/Minimum:z:0clip_by_value/y:output:0*
T0*'
_output_shapes
:€€€€€€€€€2
clip_by_value√
IdentityIdentityclip_by_value:z:0F^QNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpE^QNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp&^QNetwork/dense/BiasAdd/ReadVariableOp%^QNetwork/dense/MatMul/ReadVariableOp*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
::::2О
EQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOpEQNetwork/EncodingNetwork/EncodingNetwork/dense/BiasAdd/ReadVariableOp2М
DQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOpDQNetwork/EncodingNetwork/EncodingNetwork/dense/MatMul/ReadVariableOp2N
%QNetwork/dense/BiasAdd/ReadVariableOp%QNetwork/dense/BiasAdd/ReadVariableOp2L
$QNetwork/dense/MatMul/ReadVariableOp$QNetwork/dense/MatMul/ReadVariableOp:) %
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step
Ў

…
*__inference_polymorphic_action_fn_30108418
time_step_step_type
time_step_reward
time_step_discount
time_step_observation"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identityИҐStatefulPartitionedCallк
StatefulPartitionedCallStatefulPartitionedCalltime_step_step_typetime_step_rewardtime_step_discounttime_step_observationstatefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*
fR
__inference_action_2642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
::::22
StatefulPartitionedCallStatefulPartitionedCall:3 /
-
_user_specified_nametime_step/step_type:0,
*
_user_specified_nametime_step/reward:2.
,
_user_specified_nametime_step/discount:51
/
_user_specified_nametime_step/observation
т	
©
*__inference_polymorphic_action_fn_30108362
	time_step
time_step_1
time_step_2
time_step_3"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identityИҐStatefulPartitionedCall 
StatefulPartitionedCallStatefulPartitionedCall	time_steptime_step_1time_step_2time_step_3statefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*
fR
__inference_action_2642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step:)%
#
_user_specified_name	time_step
ъ	
Э
&__inference_signature_wrapper_30108382
discount
observation

reward
	step_type"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identityИҐStatefulPartitionedCallЎ
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationstatefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*5
f0R.
,__inference_function_with_signature_301083692
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:€€€€€€€€€:€€€€€€€€€
:€€€€€€€€€:€€€€€€€€€::::22
StatefulPartitionedCallStatefulPartitionedCall:* &
$
_user_specified_name
0/discount:-)
'
_user_specified_name0/observation:($
"
_user_specified_name
0/reward:+'
%
_user_specified_name0/step_type
џ	
Ь
%__inference_polymorphic_action_fn_271
	step_type

reward
discount
observation"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identityИҐStatefulPartitionedCall¬
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationstatefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*
fR
__inference_action_2642
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
::::22
StatefulPartitionedCallStatefulPartitionedCall:) %
#
_user_specified_name	step_type:&"
 
_user_specified_namereward:($
"
_user_specified_name
discount:+'
%
_user_specified_nameobservation
А
R
&__inference_signature_wrapper_30108405
partitionedcall_args_0
identityЦ
PartitionedCallPartitionedCallpartitionedcall_args_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*5
f0R.
,__inference_function_with_signature_301083992
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
Б
>
,__inference_function_with_signature_30108389

batch_sizeы
PartitionedFunctionCallPartitionedCall
batch_size*
Tin
2*

Tout
 *,
_gradient_op_typePartitionedCallUnused*
_output_shapes
 *
config_proto	RRШ*/
f*R(
&__inference_get_initial_state_301083882
PartitionedFunctionCall*
_input_shapes
: :* &
$
_user_specified_name
batch_size
}
8
&__inference_get_initial_state_30108388

batch_size*
_input_shapes
: :* &
$
_user_specified_name
batch_size
x
3
!__inference_get_initial_state_195

batch_size*
_input_shapes
: :* &
$
_user_specified_name
batch_size
д
Ъ
!__inference__traced_save_30108460
file_prefix4
0savev2_qnetwork_dense_kernel_read_readvariableop2
.savev2_qnetwork_dense_bias_read_readvariableopT
Psavev2_qnetwork_encodingnetwork_encodingnetwork_dense_kernel_read_readvariableopR
Nsavev2_qnetwork_encodingnetwork_encodingnetwork_dense_bias_read_readvariableop
savev2_1_const_1

identity_1ИҐMergeV2CheckpointsҐSaveV2ҐSaveV2_1•
StringJoin/inputs_1Const"/device:CPU:0*
_output_shapes
: *
dtype0*<
value3B1 B+_temp_a3f2082198a94db3808decc7e8662271/part2
StringJoin/inputs_1Б

StringJoin
StringJoinfile_prefixStringJoin/inputs_1:output:0"/device:CPU:0*
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¶
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename°
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*≥
value©B¶BK_wrapped_policy/_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEBI_wrapped_policy/_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUEBA_wrapped_policy/_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUEBA_wrapped_policy/_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUE2
SaveV2/tensor_namesР
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
SaveV2/shape_and_slices≥
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:00savev2_qnetwork_dense_kernel_read_readvariableop.savev2_qnetwork_dense_bias_read_readvariableopPsavev2_qnetwork_encodingnetwork_encodingnetwork_dense_kernel_read_readvariableopNsavev2_qnetwork_encodingnetwork_encodingnetwork_dense_bias_read_readvariableop"/device:CPU:0*
_output_shapes
 *
dtypes
22
SaveV2Г
ShardedFilename_1/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B :2
ShardedFilename_1/shardђ
ShardedFilename_1ShardedFilenameStringJoin:output:0 ShardedFilename_1/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename_1Ґ
SaveV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2_1/tensor_namesО
SaveV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
SaveV2_1/shape_and_slices—
SaveV2_1SaveV2ShardedFilename_1:filename:0SaveV2_1/tensor_names:output:0"SaveV2_1/shape_and_slices:output:0savev2_1_const_1^SaveV2"/device:CPU:0*
_output_shapes
 *
dtypes
22

SaveV2_1г
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0ShardedFilename_1:filename:0^SaveV2	^SaveV2_1"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesђ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix	^SaveV2_1"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

IdentityБ

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints^SaveV2	^SaveV2_1*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*8
_input_shapes'
%: :d::	Дd:d: 2(
MergeV2CheckpointsMergeV2Checkpoints2
SaveV2SaveV22
SaveV2_1SaveV2_1:+ '
%
_user_specified_namefile_prefix
є
5
__inference_<lambda>_201
unknown
identityJ
IdentityIdentityunknown*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
т
X
,__inference_function_with_signature_30108399
partitionedcall_args_0
identityВ
PartitionedCallPartitionedCallpartitionedcall_args_0*
Tin
2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*
_output_shapes
: *-
config_proto

CPU

GPU2*0J 8*!
fR
__inference_<lambda>_2012
PartitionedCall[
IdentityIdentityPartitionedCall:output:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
: 
ю	
£
,__inference_function_with_signature_30108369
	step_type

reward
discount
observation"
statefulpartitionedcall_args_4"
statefulpartitionedcall_args_5"
statefulpartitionedcall_args_6"
statefulpartitionedcall_args_7
identityИҐStatefulPartitionedCall÷
StatefulPartitionedCallStatefulPartitionedCall	step_typerewarddiscountobservationstatefulpartitionedcall_args_4statefulpartitionedcall_args_5statefulpartitionedcall_args_6statefulpartitionedcall_args_7*
Tin

2*
Tout
2*,
_gradient_op_typePartitionedCallUnused*'
_output_shapes
:€€€€€€€€€*-
config_proto

CPU

GPU2*0J 8*3
f.R,
*__inference_polymorphic_action_fn_301083622
StatefulPartitionedCallО
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:€€€€€€€€€2

Identity"
identityIdentity:output:0*g
_input_shapesV
T:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€:€€€€€€€€€
::::22
StatefulPartitionedCallStatefulPartitionedCall:+ '
%
_user_specified_name0/step_type:($
"
_user_specified_name
0/reward:*&
$
_user_specified_name
0/discount:-)
'
_user_specified_name0/observation
х
®
$__inference__traced_restore_30108484
file_prefix*
&assignvariableop_qnetwork_dense_kernel*
&assignvariableop_1_qnetwork_dense_biasL
Hassignvariableop_2_qnetwork_encodingnetwork_encodingnetwork_dense_kernelJ
Fassignvariableop_3_qnetwork_encodingnetwork_encodingnetwork_dense_bias

identity_5ИҐAssignVariableOpҐAssignVariableOp_1ҐAssignVariableOp_2ҐAssignVariableOp_3Ґ	RestoreV2ҐRestoreV2_1І
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*≥
value©B¶BK_wrapped_policy/_q_network/_q_value_layer/kernel/.ATTRIBUTES/VARIABLE_VALUEBI_wrapped_policy/_q_network/_q_value_layer/bias/.ATTRIBUTES/VARIABLE_VALUEBA_wrapped_policy/_q_network/variables/0/.ATTRIBUTES/VARIABLE_VALUEBA_wrapped_policy/_q_network/variables/1/.ATTRIBUTES/VARIABLE_VALUE2
RestoreV2/tensor_namesЦ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B 2
RestoreV2/shape_and_slicesњ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*$
_output_shapes
::::*
dtypes
22
	RestoreV2X
IdentityIdentityRestoreV2:tensors:0*
T0*
_output_shapes
:2

IdentityЦ
AssignVariableOpAssignVariableOp&assignvariableop_qnetwork_dense_kernelIdentity:output:0*
_output_shapes
 *
dtype02
AssignVariableOp\

Identity_1IdentityRestoreV2:tensors:1*
T0*
_output_shapes
:2

Identity_1Ь
AssignVariableOp_1AssignVariableOp&assignvariableop_1_qnetwork_dense_biasIdentity_1:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_1\

Identity_2IdentityRestoreV2:tensors:2*
T0*
_output_shapes
:2

Identity_2Њ
AssignVariableOp_2AssignVariableOpHassignvariableop_2_qnetwork_encodingnetwork_encodingnetwork_dense_kernelIdentity_2:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_2\

Identity_3IdentityRestoreV2:tensors:3*
T0*
_output_shapes
:2

Identity_3Љ
AssignVariableOp_3AssignVariableOpFassignvariableop_3_qnetwork_encodingnetwork_encodingnetwork_dense_biasIdentity_3:output:0*
_output_shapes
 *
dtype02
AssignVariableOp_3®
RestoreV2_1/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*1
value(B&B_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2_1/tensor_namesФ
RestoreV2_1/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueB
B 2
RestoreV2_1/shape_and_slicesƒ
RestoreV2_1	RestoreV2file_prefix!RestoreV2_1/tensor_names:output:0%RestoreV2_1/shape_and_slices:output:0
^RestoreV2"/device:CPU:0*
_output_shapes
:*
dtypes
22
RestoreV2_19
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpЇ

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4∆

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3
^RestoreV2^RestoreV2_1*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32
	RestoreV2	RestoreV22
RestoreV2_1RestoreV2_1:+ '
%
_user_specified_namefile_prefix
Б
8
&__inference_signature_wrapper_30108393

batch_sizeБ
PartitionedFunctionCallPartitionedCall
batch_size*
Tin
2*

Tout
 *,
_gradient_op_typePartitionedCallUnused*
_output_shapes
 *
config_proto	RRШ*5
f0R.
,__inference_function_with_signature_301083892
PartitionedFunctionCall*
_input_shapes
: :* &
$
_user_specified_name
batch_size"ѓL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*«
actionЉ
4

0/discount&
action_0/discount:0€€€€€€€€€
B
0/observation1
action_0/observation:0€€€€€€€€€

0
0/reward$
action_0/reward:0€€€€€€€€€
6
0/step_type'
action_0/step_type:0€€€€€€€€€:
action0
StatefulPartitionedCall:0€€€€€€€€€tensorflow/serving/predict*e
get_initial_stateP
2

batch_size$
get_initial_state_batch_size:0 tensorflow/serving/predict*P
get_train_step> 
int32
PartitionedCall:0 tensorflow/serving/predict: S
v
_wrapped_policy

signatures

5action
6get_initial_state
7
train_step"
_generic_user_object
.

_q_network"
_generic_user_object
N

8action
9get_initial_state
:get_train_step"
signature_map
ч
_encoder
_q_value_layer
	variables
trainable_variables
regularization_losses
		keras_api
;__call__
*<&call_and_return_all_conditional_losses"∆
_tf_keras_network™{"class_name": "QNetwork", "name": "QNetwork", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false}
ю

_postprocessing_layers
	variables
trainable_variables
regularization_losses
	keras_api
=__call__
*>&call_and_return_all_conditional_losses"”
_tf_keras_networkЈ{"class_name": "EncodingNetwork", "name": "EncodingNetwork", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "is_graph_network": false}
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
*@&call_and_return_all_conditional_losses"£
_tf_keras_layerЙ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 5, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "RandomUniform", "config": {"minval": -0.03, "maxval": 0.03, "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Constant", "config": {"value": -0.2, "dtype": "float32"}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 100}}}}
<
0
1
2
3"
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
	variables
non_trainable_variables
layer_regularization_losses
trainable_variables

layers
metrics
regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
	variables
non_trainable_variables
layer_regularization_losses
trainable_variables

layers
 metrics
regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
':%d2QNetwork/dense/kernel
!:2QNetwork/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
	variables
!non_trainable_variables
"layer_regularization_losses
trainable_variables

#layers
$metrics
regularization_losses
?__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
H:F	Дd25QNetwork/EncodingNetwork/EncodingNetwork/dense/kernel
A:?d23QNetwork/EncodingNetwork/EncodingNetwork/dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
ђ
%	variables
&trainable_variables
'regularization_losses
(	keras_api
A__call__
*B&call_and_return_all_conditional_losses"Э
_tf_keras_layerГ{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
й

kernel
bias
)	variables
*trainable_variables
+regularization_losses
,	keras_api
C__call__
*D&call_and_return_all_conditional_losses"ƒ
_tf_keras_layer™{"class_name": "Dense", "name": "EncodingNetwork/dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "config": {"name": "EncodingNetwork/dense", "trainable": true, "dtype": "float32", "units": 100, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "VarianceScaling", "config": {"scale": 2.0, "mode": "fan_in", "distribution": "truncated_normal", "seed": null, "dtype": "float32"}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 260}}}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
%	variables
-non_trainable_variables
.layer_regularization_losses
&trainable_variables

/layers
0metrics
'regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
Ъ
)	variables
1non_trainable_variables
2layer_regularization_losses
*trainable_variables

3layers
4metrics
+regularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
К2З
%__inference_polymorphic_action_fn_271
*__inference_polymorphic_action_fn_30108418±
™≤¶
FullArgSpec(
args Ъ
j	time_step
jpolicy_state
varargs
 
varkw
 
defaultsҐ
Ґ 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ѕ2ћ
!__inference_get_initial_state_195¶
Э≤Щ
FullArgSpec!
argsЪ
jself
j
batch_size
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
B
__inference_<lambda>_201
\BZ
&__inference_signature_wrapper_30108382
0/discount0/observation0/reward0/step_type
8B6
&__inference_signature_wrapper_30108393
batch_size
*B(
&__inference_signature_wrapper_30108405
е2вя
÷≤“
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
е2вя
÷≤“
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
ж2га
„≤”
FullArgSpecL
argsDЪA
jself
jobservation
j	step_type
jnetwork_state

jtraining
varargs
 
varkw
 
defaultsЪ

 
Ґ 
p 

kwonlyargsЪ 
kwonlydefaults™ 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
®2•Ґ
Щ≤Х
FullArgSpec
argsЪ
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotations™ *
 
	J
Const7
__inference_<lambda>_201EҐ

Ґ 
™ "К N
!__inference_get_initial_state_195)"Ґ
Ґ
К

batch_size 
™ "Ґ н
%__inference_polymorphic_action_fn_271√вҐё
÷Ґ“
 ≤∆
TimeStep,
	step_typeК
	step_type€€€€€€€€€&
rewardК
reward€€€€€€€€€*
discountК
discount€€€€€€€€€8
observation)К&
observation€€€€€€€€€

Ґ 
™ "V≤S

PolicyStep*
action К
action€€€€€€€€€
stateҐ 
infoҐ Ъ
*__inference_polymorphic_action_fn_30108418лКҐЖ
юҐъ
т≤о
TimeStep6
	step_type)К&
time_step/step_type€€€€€€€€€0
reward&К#
time_step/reward€€€€€€€€€4
discount(К%
time_step/discount€€€€€€€€€B
observation3К0
time_step/observation€€€€€€€€€

Ґ 
™ "V≤S

PolicyStep*
action К
action€€€€€€€€€
stateҐ 
infoҐ Ѕ
&__inference_signature_wrapper_30108382Ц№ҐЎ
Ґ 
–™ћ
.

0/discount К

0/discount€€€€€€€€€
<
0/observation+К(
0/observation€€€€€€€€€

*
0/rewardК
0/reward€€€€€€€€€
0
0/step_type!К
0/step_type€€€€€€€€€"/™,
*
action К
action€€€€€€€€€a
&__inference_signature_wrapper_3010839370Ґ-
Ґ 
&™#
!

batch_sizeК

batch_size "™ Z
&__inference_signature_wrapper_301084050EҐ

Ґ 
™ "™

int32К
int32 