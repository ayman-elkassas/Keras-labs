Κο
Ή
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
Α
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
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.02v2.8.0-rc1-32-g3f878cff5b68Ζ

sequential/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namesequential/dense/kernel

+sequential/dense/kernel/Read/ReadVariableOpReadVariableOpsequential/dense/kernel* 
_output_shapes
:
*
dtype0

sequential/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*&
shared_namesequential/dense/bias
|
)sequential/dense/bias/Read/ReadVariableOpReadVariableOpsequential/dense/bias*
_output_shapes	
:*
dtype0

sequential/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
**
shared_namesequential/dense_1/kernel

-sequential/dense_1/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_1/kernel* 
_output_shapes
:
*
dtype0

sequential/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namesequential/dense_1/bias

+sequential/dense_1/bias/Read/ReadVariableOpReadVariableOpsequential/dense_1/bias*
_output_shapes	
:*
dtype0

sequential/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
**
shared_namesequential/dense_2/kernel

-sequential/dense_2/kernel/Read/ReadVariableOpReadVariableOpsequential/dense_2/kernel*
_output_shapes
:	
*
dtype0

sequential/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*(
shared_namesequential/dense_2/bias

+sequential/dense_2/bias/Read/ReadVariableOpReadVariableOpsequential/dense_2/bias*
_output_shapes
:
*
dtype0
x
training/Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *#
shared_nametraining/Adam/iter
q
&training/Adam/iter/Read/ReadVariableOpReadVariableOptraining/Adam/iter*
_output_shapes
: *
dtype0	
|
training/Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_1
u
(training/Adam/beta_1/Read/ReadVariableOpReadVariableOptraining/Adam/beta_1*
_output_shapes
: *
dtype0
|
training/Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *%
shared_nametraining/Adam/beta_2
u
(training/Adam/beta_2/Read/ReadVariableOpReadVariableOptraining/Adam/beta_2*
_output_shapes
: *
dtype0
z
training/Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *$
shared_nametraining/Adam/decay
s
'training/Adam/decay/Read/ReadVariableOpReadVariableOptraining/Adam/decay*
_output_shapes
: *
dtype0

training/Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_nametraining/Adam/learning_rate

/training/Adam/learning_rate/Read/ReadVariableOpReadVariableOptraining/Adam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
¬
'training/Adam/sequential/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'training/Adam/sequential/dense/kernel/m
₯
;training/Adam/sequential/dense/kernel/m/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense/kernel/m* 
_output_shapes
:
*
dtype0
£
%training/Adam/sequential/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%training/Adam/sequential/dense/bias/m

9training/Adam/sequential/dense/bias/m/Read/ReadVariableOpReadVariableOp%training/Adam/sequential/dense/bias/m*
_output_shapes	
:*
dtype0
°
)training/Adam/sequential/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)training/Adam/sequential/dense_1/kernel/m
©
=training/Adam/sequential/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_1/kernel/m* 
_output_shapes
:
*
dtype0
§
'training/Adam/sequential/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'training/Adam/sequential/dense_1/bias/m
 
;training/Adam/sequential/dense_1/bias/m/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_1/bias/m*
_output_shapes	
:*
dtype0
―
)training/Adam/sequential/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*:
shared_name+)training/Adam/sequential/dense_2/kernel/m
¨
=training/Adam/sequential/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_2/kernel/m*
_output_shapes
:	
*
dtype0
¦
'training/Adam/sequential/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'training/Adam/sequential/dense_2/bias/m

;training/Adam/sequential/dense_2/bias/m/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_2/bias/m*
_output_shapes
:
*
dtype0
¬
'training/Adam/sequential/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'training/Adam/sequential/dense/kernel/v
₯
;training/Adam/sequential/dense/kernel/v/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense/kernel/v* 
_output_shapes
:
*
dtype0
£
%training/Adam/sequential/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%training/Adam/sequential/dense/bias/v

9training/Adam/sequential/dense/bias/v/Read/ReadVariableOpReadVariableOp%training/Adam/sequential/dense/bias/v*
_output_shapes	
:*
dtype0
°
)training/Adam/sequential/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*:
shared_name+)training/Adam/sequential/dense_1/kernel/v
©
=training/Adam/sequential/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_1/kernel/v* 
_output_shapes
:
*
dtype0
§
'training/Adam/sequential/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*8
shared_name)'training/Adam/sequential/dense_1/bias/v
 
;training/Adam/sequential/dense_1/bias/v/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_1/bias/v*
_output_shapes	
:*
dtype0
―
)training/Adam/sequential/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	
*:
shared_name+)training/Adam/sequential/dense_2/kernel/v
¨
=training/Adam/sequential/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp)training/Adam/sequential/dense_2/kernel/v*
_output_shapes
:	
*
dtype0
¦
'training/Adam/sequential/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*8
shared_name)'training/Adam/sequential/dense_2/bias/v

;training/Adam/sequential/dense_2/bias/v/Read/ReadVariableOpReadVariableOp'training/Adam/sequential/dense_2/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
¨1
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*γ0
valueΩ0BΦ0 BΟ0
Ξ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses* 
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses*
¦

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses*
°
,iter

-beta_1

.beta_2
	/decay
0learning_ratemQmRmSmT$mU%mVvWvXvYvZ$v[%v\*
.
0
1
2
3
$4
%5*
.
0
1
2
3
$4
%5*
* 
°
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
* 

6serving_default* 
* 
* 
* 

7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 
* 
* 
ga
VARIABLE_VALUEsequential/dense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
c]
VARIABLE_VALUEsequential/dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
ic
VARIABLE_VALUEsequential/dense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*
* 
* 
ic
VARIABLE_VALUEsequential/dense_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
e_
VARIABLE_VALUEsequential/dense_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

$0
%1*

$0
%1*
* 

Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses*
* 
* 
UO
VARIABLE_VALUEtraining/Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining/Adam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
YS
VARIABLE_VALUEtraining/Adam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
WQ
VARIABLE_VALUEtraining/Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
ga
VARIABLE_VALUEtraining/Adam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 
 
0
1
2
3*

K0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
H
	Ltotal
	Mcount
N
_fn_kwargs
O	variables
P	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*
* 

L0
M1*

O	variables*

VARIABLE_VALUE'training/Adam/sequential/dense/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%training/Adam/sequential/dense/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)training/Adam/sequential/dense_1/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'training/Adam/sequential/dense_1/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)training/Adam/sequential/dense_2/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'training/Adam/sequential/dense_2/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'training/Adam/sequential/dense/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE%training/Adam/sequential/dense/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)training/Adam/sequential/dense_1/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'training/Adam/sequential/dense_1/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE)training/Adam/sequential/dense_2/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

VARIABLE_VALUE'training/Adam/sequential/dense_2/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*

serving_default_input_1Placeholder*+
_output_shapes
:?????????*
dtype0* 
shape:?????????
Σ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1sequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference_signature_wrapper_692
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
­
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename+sequential/dense/kernel/Read/ReadVariableOp)sequential/dense/bias/Read/ReadVariableOp-sequential/dense_1/kernel/Read/ReadVariableOp+sequential/dense_1/bias/Read/ReadVariableOp-sequential/dense_2/kernel/Read/ReadVariableOp+sequential/dense_2/bias/Read/ReadVariableOp&training/Adam/iter/Read/ReadVariableOp(training/Adam/beta_1/Read/ReadVariableOp(training/Adam/beta_2/Read/ReadVariableOp'training/Adam/decay/Read/ReadVariableOp/training/Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp;training/Adam/sequential/dense/kernel/m/Read/ReadVariableOp9training/Adam/sequential/dense/bias/m/Read/ReadVariableOp=training/Adam/sequential/dense_1/kernel/m/Read/ReadVariableOp;training/Adam/sequential/dense_1/bias/m/Read/ReadVariableOp=training/Adam/sequential/dense_2/kernel/m/Read/ReadVariableOp;training/Adam/sequential/dense_2/bias/m/Read/ReadVariableOp;training/Adam/sequential/dense/kernel/v/Read/ReadVariableOp9training/Adam/sequential/dense/bias/v/Read/ReadVariableOp=training/Adam/sequential/dense_1/kernel/v/Read/ReadVariableOp;training/Adam/sequential/dense_1/bias/v/Read/ReadVariableOp=training/Adam/sequential/dense_2/kernel/v/Read/ReadVariableOp;training/Adam/sequential/dense_2/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *%
f R
__inference__traced_save_855
΄
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamesequential/dense/kernelsequential/dense/biassequential/dense_1/kernelsequential/dense_1/biassequential/dense_2/kernelsequential/dense_2/biastraining/Adam/itertraining/Adam/beta_1training/Adam/beta_2training/Adam/decaytraining/Adam/learning_ratetotalcount'training/Adam/sequential/dense/kernel/m%training/Adam/sequential/dense/bias/m)training/Adam/sequential/dense_1/kernel/m'training/Adam/sequential/dense_1/bias/m)training/Adam/sequential/dense_2/kernel/m'training/Adam/sequential/dense_2/bias/m'training/Adam/sequential/dense/kernel/v%training/Adam/sequential/dense/bias/v)training/Adam/sequential/dense_1/kernel/v'training/Adam/sequential/dense_1/bias/v)training/Adam/sequential/dense_2/kernel/v'training/Adam/sequential/dense_2/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_restore_940Τύ
ύ	
ΰ
(__inference_sequential_layer_call_fn_612

inputs+
sequential_dense_kernel:
$
sequential_dense_bias:	-
sequential_dense_1_kernel:
&
sequential_dense_1_bias:	,
sequential_dense_2_kernel:	
%
sequential_dense_2_bias:

identity’StatefulPartitionedCallδ
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_415o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Τ	
Ϊ
!__inference_signature_wrapper_692
input_1+
sequential_dense_kernel:
$
sequential_dense_bias:	-
sequential_dense_1_kernel:
&
sequential_dense_1_bias:	,
sequential_dense_2_kernel:	
%
sequential_dense_2_bias:

identity’StatefulPartitionedCallΐ
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *'
f"R 
__inference__wrapped_model_353o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
·
ς
C__inference_sequential_layer_call_and_return_conditional_losses_586
input_11
dense_sequential_dense_kernel:
*
dense_sequential_dense_bias:	5
!dense_1_sequential_dense_1_kernel:
.
dense_1_sequential_dense_1_bias:	4
!dense_2_sequential_dense_2_kernel:	
-
dense_2_sequential_dense_2_bias:

identity’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCallb
flatten/CastCastinput_1*

DstT0*

SrcT0*+
_output_shapes
:?????????Ύ
flatten/PartitionedCallPartitionedCallflatten/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_367
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_sequential_dense_kerneldense_sequential_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_380?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0!dense_1_sequential_dense_1_kerneldense_1_sequential_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_395―
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0!dense_2_sequential_dense_2_kerneldense_2_sequential_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_410w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
ͺ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
±


>__inference_dense_layer_call_and_return_conditional_losses_721

inputsA
-matmul_readvariableop_sequential_dense_kernel:
;
,biasadd_readvariableop_sequential_dense_bias:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_sequential_dense_kernel* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_sequential_dense_bias*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Ί
\
@__inference_flatten_layer_call_and_return_conditional_losses_367

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs


α
(__inference_sequential_layer_call_fn_424
input_1+
sequential_dense_kernel:
$
sequential_dense_bias:	-
sequential_dense_1_kernel:
&
sequential_dense_1_bias:	,
sequential_dense_2_kernel:	
%
sequential_dense_2_bias:

identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_415o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
»


@__inference_dense_1_layer_call_and_return_conditional_losses_739

inputsC
/matmul_readvariableop_sequential_dense_1_kernel:
=
.biasadd_readvariableop_sequential_dense_1_bias:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_1_kernel* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ε


@__inference_dense_1_layer_call_and_return_conditional_losses_395

inputsC
/matmul_readvariableop_sequential_dense_1_kernel:
=
.biasadd_readvariableop_sequential_dense_1_bias:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_1_kernel* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
Λ
―
#__inference_dense_layer_call_fn_710

inputs+
sequential_dense_kernel:
$
sequential_dense_bias:	
identity’StatefulPartitionedCallπ
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_kernelsequential_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_380p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ή
ρ
C__inference_sequential_layer_call_and_return_conditional_losses_521

inputs1
dense_sequential_dense_kernel:
*
dense_sequential_dense_bias:	5
!dense_1_sequential_dense_1_kernel:
.
dense_1_sequential_dense_1_bias:	4
!dense_2_sequential_dense_2_kernel:	
-
dense_2_sequential_dense_2_bias:

identity’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCalla
flatten/CastCastinputs*

DstT0*

SrcT0*+
_output_shapes
:?????????Ύ
flatten/PartitionedCallPartitionedCallflatten/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_367
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_sequential_dense_kerneldense_sequential_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_380?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0!dense_1_sequential_dense_1_kerneldense_1_sequential_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_395―
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0!dense_2_sequential_dense_2_kerneldense_2_sequential_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_410w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
ͺ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
·
ς
C__inference_sequential_layer_call_and_return_conditional_losses_601
input_11
dense_sequential_dense_kernel:
*
dense_sequential_dense_bias:	5
!dense_1_sequential_dense_1_kernel:
.
dense_1_sequential_dense_1_bias:	4
!dense_2_sequential_dense_2_kernel:	
-
dense_2_sequential_dense_2_bias:

identity’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCallb
flatten/CastCastinput_1*

DstT0*

SrcT0*+
_output_shapes
:?????????Ύ
flatten/PartitionedCallPartitionedCallflatten/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_367
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_sequential_dense_kerneldense_sequential_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_380?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0!dense_1_sequential_dense_1_kerneldense_1_sequential_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_395―
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0!dense_2_sequential_dense_2_kerneldense_2_sequential_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_410w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
ͺ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
Χ
΅
%__inference_dense_1_layer_call_fn_728

inputs-
sequential_dense_1_kernel:
&
sequential_dense_1_bias:	
identity’StatefulPartitionedCallφ
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_1_kernelsequential_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_395p
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*(
_output_shapes
:?????????`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs


α
(__inference_sequential_layer_call_fn_571
input_1+
sequential_dense_kernel:
$
sequential_dense_bias:	-
sequential_dense_1_kernel:
&
sequential_dense_1_bias:	,
sequential_dense_2_kernel:	
%
sequential_dense_2_bias:

identity’StatefulPartitionedCallε
StatefulPartitionedCallStatefulPartitionedCallinput_1sequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_521o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
ύ	
ΰ
(__inference_sequential_layer_call_fn_623

inputs+
sequential_dense_kernel:
$
sequential_dense_bias:	-
sequential_dense_1_kernel:
&
sequential_dense_1_bias:	,
sequential_dense_2_kernel:	
%
sequential_dense_2_bias:

identity’StatefulPartitionedCallδ
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_kernelsequential_dense_biassequential_dense_1_kernelsequential_dense_1_biassequential_dense_2_kernelsequential_dense_2_bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_sequential_layer_call_and_return_conditional_losses_521o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
§%
³
__inference__wrapped_model_353
input_1R
>sequential_dense_matmul_readvariableop_sequential_dense_kernel:
L
=sequential_dense_biasadd_readvariableop_sequential_dense_bias:	V
Bsequential_dense_1_matmul_readvariableop_sequential_dense_1_kernel:
P
Asequential_dense_1_biasadd_readvariableop_sequential_dense_1_bias:	U
Bsequential_dense_2_matmul_readvariableop_sequential_dense_2_kernel:	
O
Asequential_dense_2_biasadd_readvariableop_sequential_dense_2_bias:

identity’'sequential/dense/BiasAdd/ReadVariableOp’&sequential/dense/MatMul/ReadVariableOp’)sequential/dense_1/BiasAdd/ReadVariableOp’(sequential/dense_1/MatMul/ReadVariableOp’)sequential/dense_2/BiasAdd/ReadVariableOp’(sequential/dense_2/MatMul/ReadVariableOpm
sequential/flatten/CastCastinput_1*

DstT0*

SrcT0*+
_output_shapes
:?????????i
sequential/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  
sequential/flatten/ReshapeReshapesequential/flatten/Cast:y:0!sequential/flatten/Const:output:0*
T0*(
_output_shapes
:?????????§
&sequential/dense/MatMul/ReadVariableOpReadVariableOp>sequential_dense_matmul_readvariableop_sequential_dense_kernel* 
_output_shapes
:
*
dtype0©
sequential/dense/MatMulMatMul#sequential/flatten/Reshape:output:0.sequential/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????’
'sequential/dense/BiasAdd/ReadVariableOpReadVariableOp=sequential_dense_biasadd_readvariableop_sequential_dense_bias*
_output_shapes	
:*
dtype0ͺ
sequential/dense/BiasAddBiasAdd!sequential/dense/MatMul:product:0/sequential/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????s
sequential/dense/ReluRelu!sequential/dense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????­
(sequential/dense_1/MatMul/ReadVariableOpReadVariableOpBsequential_dense_1_matmul_readvariableop_sequential_dense_1_kernel* 
_output_shapes
:
*
dtype0­
sequential/dense_1/MatMulMatMul#sequential/dense/Relu:activations:00sequential/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????¨
)sequential/dense_1/BiasAdd/ReadVariableOpReadVariableOpAsequential_dense_1_biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes	
:*
dtype0°
sequential/dense_1/BiasAddBiasAdd#sequential/dense_1/MatMul:product:01sequential/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????w
sequential/dense_1/ReluRelu#sequential/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????¬
(sequential/dense_2/MatMul/ReadVariableOpReadVariableOpBsequential_dense_2_matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes
:	
*
dtype0?
sequential/dense_2/MatMulMatMul%sequential/dense_1/Relu:activations:00sequential/dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
§
)sequential/dense_2/BiasAdd/ReadVariableOpReadVariableOpAsequential_dense_2_biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:
*
dtype0―
sequential/dense_2/BiasAddBiasAdd#sequential/dense_2/MatMul:product:01sequential/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
|
sequential/dense_2/SoftmaxSoftmax#sequential/dense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
s
IdentityIdentity$sequential/dense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
Η
NoOpNoOp(^sequential/dense/BiasAdd/ReadVariableOp'^sequential/dense/MatMul/ReadVariableOp*^sequential/dense_1/BiasAdd/ReadVariableOp)^sequential/dense_1/MatMul/ReadVariableOp*^sequential/dense_2/BiasAdd/ReadVariableOp)^sequential/dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 2R
'sequential/dense/BiasAdd/ReadVariableOp'sequential/dense/BiasAdd/ReadVariableOp2P
&sequential/dense/MatMul/ReadVariableOp&sequential/dense/MatMul/ReadVariableOp2V
)sequential/dense_1/BiasAdd/ReadVariableOp)sequential/dense_1/BiasAdd/ReadVariableOp2T
(sequential/dense_1/MatMul/ReadVariableOp(sequential/dense_1/MatMul/ReadVariableOp2V
)sequential/dense_2/BiasAdd/ReadVariableOp)sequential/dense_2/BiasAdd/ReadVariableOp2T
(sequential/dense_2/MatMul/ReadVariableOp(sequential/dense_2/MatMul/ReadVariableOp:T P
+
_output_shapes
:?????????
!
_user_specified_name	input_1
Ϋ


>__inference_dense_layer_call_and_return_conditional_losses_380

inputsA
-matmul_readvariableop_sequential_dense_kernel:
;
,biasadd_readvariableop_sequential_dense_bias:	
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp-matmul_readvariableop_sequential_dense_kernel* 
_output_shapes
:
*
dtype0j
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
BiasAdd/ReadVariableOpReadVariableOp,biasadd_readvariableop_sequential_dense_bias*
_output_shapes	
:*
dtype0w
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????Q
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:?????????b
IdentityIdentityRelu:activations:0^NoOp*
T0*(
_output_shapes
:?????????w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
β


@__inference_dense_2_layer_call_and_return_conditional_losses_410

inputsB
/matmul_readvariableop_sequential_dense_2_kernel:	
<
.biasadd_readvariableop_sequential_dense_2_bias:

identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
υ
A
%__inference_flatten_layer_call_fn_697

inputs
identity¬
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_367a
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs

\
@__inference_flatten_layer_call_and_return_conditional_losses_703

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"????  ]
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????Y
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????"
identityIdentity:output:0**
_input_shapes
:?????????:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Έ


@__inference_dense_2_layer_call_and_return_conditional_losses_757

inputsB
/matmul_readvariableop_sequential_dense_2_kernel:	
<
.biasadd_readvariableop_sequential_dense_2_bias:

identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOp
MatMul/ReadVariableOpReadVariableOp/matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes
:	
*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

BiasAdd/ReadVariableOpReadVariableOp.biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:
*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
V
SoftmaxSoftmaxBiasAdd:output:0*
T0*'
_output_shapes
:?????????
`
IdentityIdentitySoftmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????
w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
°
Σ
C__inference_sequential_layer_call_and_return_conditional_losses_679

inputsG
3dense_matmul_readvariableop_sequential_dense_kernel:
A
2dense_biasadd_readvariableop_sequential_dense_bias:	K
7dense_1_matmul_readvariableop_sequential_dense_1_kernel:
E
6dense_1_biasadd_readvariableop_sequential_dense_1_bias:	J
7dense_2_matmul_readvariableop_sequential_dense_2_kernel:	
D
6dense_2_biasadd_readvariableop_sequential_dense_2_bias:

identity’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOpa
flatten/CastCastinputs*

DstT0*

SrcT0*+
_output_shapes
:?????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  w
flatten/ReshapeReshapeflatten/Cast:y:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_sequential_dense_kernel* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_sequential_dense_bias*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_sequential_dense_1_kernel* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes
:	
*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:
*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????

NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Ή>
ψ
__inference__traced_save_855
file_prefix6
2savev2_sequential_dense_kernel_read_readvariableop4
0savev2_sequential_dense_bias_read_readvariableop8
4savev2_sequential_dense_1_kernel_read_readvariableop6
2savev2_sequential_dense_1_bias_read_readvariableop8
4savev2_sequential_dense_2_kernel_read_readvariableop6
2savev2_sequential_dense_2_bias_read_readvariableop1
-savev2_training_adam_iter_read_readvariableop	3
/savev2_training_adam_beta_1_read_readvariableop3
/savev2_training_adam_beta_2_read_readvariableop2
.savev2_training_adam_decay_read_readvariableop:
6savev2_training_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopF
Bsavev2_training_adam_sequential_dense_kernel_m_read_readvariableopD
@savev2_training_adam_sequential_dense_bias_m_read_readvariableopH
Dsavev2_training_adam_sequential_dense_1_kernel_m_read_readvariableopF
Bsavev2_training_adam_sequential_dense_1_bias_m_read_readvariableopH
Dsavev2_training_adam_sequential_dense_2_kernel_m_read_readvariableopF
Bsavev2_training_adam_sequential_dense_2_bias_m_read_readvariableopF
Bsavev2_training_adam_sequential_dense_kernel_v_read_readvariableopD
@savev2_training_adam_sequential_dense_bias_v_read_readvariableopH
Dsavev2_training_adam_sequential_dense_1_kernel_v_read_readvariableopF
Bsavev2_training_adam_sequential_dense_1_bias_v_read_readvariableopH
Dsavev2_training_adam_sequential_dense_2_kernel_v_read_readvariableopF
Bsavev2_training_adam_sequential_dense_2_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B₯B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH‘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B τ
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:02savev2_sequential_dense_kernel_read_readvariableop0savev2_sequential_dense_bias_read_readvariableop4savev2_sequential_dense_1_kernel_read_readvariableop2savev2_sequential_dense_1_bias_read_readvariableop4savev2_sequential_dense_2_kernel_read_readvariableop2savev2_sequential_dense_2_bias_read_readvariableop-savev2_training_adam_iter_read_readvariableop/savev2_training_adam_beta_1_read_readvariableop/savev2_training_adam_beta_2_read_readvariableop.savev2_training_adam_decay_read_readvariableop6savev2_training_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopBsavev2_training_adam_sequential_dense_kernel_m_read_readvariableop@savev2_training_adam_sequential_dense_bias_m_read_readvariableopDsavev2_training_adam_sequential_dense_1_kernel_m_read_readvariableopBsavev2_training_adam_sequential_dense_1_bias_m_read_readvariableopDsavev2_training_adam_sequential_dense_2_kernel_m_read_readvariableopBsavev2_training_adam_sequential_dense_2_bias_m_read_readvariableopBsavev2_training_adam_sequential_dense_kernel_v_read_readvariableop@savev2_training_adam_sequential_dense_bias_v_read_readvariableopDsavev2_training_adam_sequential_dense_1_kernel_v_read_readvariableopBsavev2_training_adam_sequential_dense_1_bias_v_read_readvariableopDsavev2_training_adam_sequential_dense_2_kernel_v_read_readvariableopBsavev2_training_adam_sequential_dense_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*Μ
_input_shapesΊ
·: :
::
::	
:
: : : : : : : :
::
::	
:
:
::
::	
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	
: 

_output_shapes
:
:

_output_shapes
: 
Σ
³
%__inference_dense_2_layer_call_fn_746

inputs,
sequential_dense_2_kernel:	
%
sequential_dense_2_bias:

identity’StatefulPartitionedCallυ
StatefulPartitionedCallStatefulPartitionedCallinputssequential_dense_2_kernelsequential_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_410o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*+
_input_shapes
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????
 
_user_specified_nameinputs
ή
ρ
C__inference_sequential_layer_call_and_return_conditional_losses_415

inputs1
dense_sequential_dense_kernel:
*
dense_sequential_dense_bias:	5
!dense_1_sequential_dense_1_kernel:
.
dense_1_sequential_dense_1_bias:	4
!dense_2_sequential_dense_2_kernel:	
-
dense_2_sequential_dense_2_bias:

identity’dense/StatefulPartitionedCall’dense_1/StatefulPartitionedCall’dense_2/StatefulPartitionedCalla
flatten/CastCastinputs*

DstT0*

SrcT0*+
_output_shapes
:?????????Ύ
flatten/PartitionedCallPartitionedCallflatten/Cast:y:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_flatten_layer_call_and_return_conditional_losses_367
dense/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0dense_sequential_dense_kerneldense_sequential_dense_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *G
fBR@
>__inference_dense_layer_call_and_return_conditional_losses_380?
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0!dense_1_sequential_dense_1_kerneldense_1_sequential_dense_1_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_1_layer_call_and_return_conditional_losses_395―
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0!dense_2_sequential_dense_2_kerneldense_2_sequential_dense_2_bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *I
fDRB
@__inference_dense_2_layer_call_and_return_conditional_losses_410w
IdentityIdentity(dense_2/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????
ͺ
NoOpNoOp^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*6
_input_shapes%
#:?????????: : : : : : 2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs
Τj
Λ
__inference__traced_restore_940
file_prefix<
(assignvariableop_sequential_dense_kernel:
7
(assignvariableop_1_sequential_dense_bias:	@
,assignvariableop_2_sequential_dense_1_kernel:
9
*assignvariableop_3_sequential_dense_1_bias:	?
,assignvariableop_4_sequential_dense_2_kernel:	
8
*assignvariableop_5_sequential_dense_2_bias:
/
%assignvariableop_6_training_adam_iter:	 1
'assignvariableop_7_training_adam_beta_1: 1
'assignvariableop_8_training_adam_beta_2: 0
&assignvariableop_9_training_adam_decay: 9
/assignvariableop_10_training_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: O
;assignvariableop_13_training_adam_sequential_dense_kernel_m:
H
9assignvariableop_14_training_adam_sequential_dense_bias_m:	Q
=assignvariableop_15_training_adam_sequential_dense_1_kernel_m:
J
;assignvariableop_16_training_adam_sequential_dense_1_bias_m:	P
=assignvariableop_17_training_adam_sequential_dense_2_kernel_m:	
I
;assignvariableop_18_training_adam_sequential_dense_2_bias_m:
O
;assignvariableop_19_training_adam_sequential_dense_kernel_v:
H
9assignvariableop_20_training_adam_sequential_dense_bias_v:	Q
=assignvariableop_21_training_adam_sequential_dense_1_kernel_v:
J
;assignvariableop_22_training_adam_sequential_dense_1_bias_v:	P
=assignvariableop_23_training_adam_sequential_dense_2_kernel_v:	
I
;assignvariableop_24_training_adam_sequential_dense_2_bias_v:

identity_26’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B₯B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH€
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B  
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp(assignvariableop_sequential_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp(assignvariableop_1_sequential_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp,assignvariableop_2_sequential_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp*assignvariableop_3_sequential_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp,assignvariableop_4_sequential_dense_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp*assignvariableop_5_sequential_dense_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOp%assignvariableop_6_training_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOp'assignvariableop_7_training_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOp'assignvariableop_8_training_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOp&assignvariableop_9_training_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
: 
AssignVariableOp_10AssignVariableOp/assignvariableop_10_training_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_13AssignVariableOp;assignvariableop_13_training_adam_sequential_dense_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_14AssignVariableOp9assignvariableop_14_training_adam_sequential_dense_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_15AssignVariableOp=assignvariableop_15_training_adam_sequential_dense_1_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_16AssignVariableOp;assignvariableop_16_training_adam_sequential_dense_1_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_17AssignVariableOp=assignvariableop_17_training_adam_sequential_dense_2_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_18AssignVariableOp;assignvariableop_18_training_adam_sequential_dense_2_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_19AssignVariableOp;assignvariableop_19_training_adam_sequential_dense_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:ͺ
AssignVariableOp_20AssignVariableOp9assignvariableop_20_training_adam_sequential_dense_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_21AssignVariableOp=assignvariableop_21_training_adam_sequential_dense_1_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_22AssignVariableOp;assignvariableop_22_training_adam_sequential_dense_1_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:?
AssignVariableOp_23AssignVariableOp=assignvariableop_23_training_adam_sequential_dense_2_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:¬
AssignVariableOp_24AssignVariableOp;assignvariableop_24_training_adam_sequential_dense_2_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 υ
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: β
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
°
Σ
C__inference_sequential_layer_call_and_return_conditional_losses_651

inputsG
3dense_matmul_readvariableop_sequential_dense_kernel:
A
2dense_biasadd_readvariableop_sequential_dense_bias:	K
7dense_1_matmul_readvariableop_sequential_dense_1_kernel:
E
6dense_1_biasadd_readvariableop_sequential_dense_1_bias:	J
7dense_2_matmul_readvariableop_sequential_dense_2_kernel:	
D
6dense_2_biasadd_readvariableop_sequential_dense_2_bias:

identity’dense/BiasAdd/ReadVariableOp’dense/MatMul/ReadVariableOp’dense_1/BiasAdd/ReadVariableOp’dense_1/MatMul/ReadVariableOp’dense_2/BiasAdd/ReadVariableOp’dense_2/MatMul/ReadVariableOpa
flatten/CastCastinputs*

DstT0*

SrcT0*+
_output_shapes
:?????????^
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"????  w
flatten/ReshapeReshapeflatten/Cast:y:0flatten/Const:output:0*
T0*(
_output_shapes
:?????????
dense/MatMul/ReadVariableOpReadVariableOp3dense_matmul_readvariableop_sequential_dense_kernel* 
_output_shapes
:
*
dtype0
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense/BiasAdd/ReadVariableOpReadVariableOp2dense_biasadd_readvariableop_sequential_dense_bias*
_output_shapes	
:*
dtype0
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????]

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_1/MatMul/ReadVariableOpReadVariableOp7dense_1_matmul_readvariableop_sequential_dense_1_kernel* 
_output_shapes
:
*
dtype0
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????
dense_1/BiasAdd/ReadVariableOpReadVariableOp6dense_1_biasadd_readvariableop_sequential_dense_1_bias*
_output_shapes	
:*
dtype0
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:?????????a
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:?????????
dense_2/MatMul/ReadVariableOpReadVariableOp7dense_2_matmul_readvariableop_sequential_dense_2_kernel*
_output_shapes
:	
*
dtype0
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????

dense_2/BiasAdd/ReadVariableOpReadVariableOp6dense_2_biasadd_readvariableop_sequential_dense_2_bias*
_output_shapes
:
*
dtype0
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
f
dense_2/SoftmaxSoftmaxdense_2/BiasAdd:output:0*
T0*'
_output_shapes
:?????????
h
IdentityIdentitydense_2/Softmax:softmax:0^NoOp*
T0*'
_output_shapes
:?????????

NoOpNoOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*6
_input_shapes%
#:?????????: : : : : : 2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:S O
+
_output_shapes
:?????????
 
_user_specified_nameinputs"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*―
serving_default
?
input_14
serving_default_input_1:0?????????<
output_10
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:ΰZ
θ
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
	optimizer
	variables
trainable_variables
regularization_losses
		keras_api

__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
₯
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses"
_tf_keras_layer
»

$kernel
%bias
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
Ώ
,iter

-beta_1

.beta_2
	/decay
0learning_ratemQmRmSmT$mU%mVvWvXvYvZ$v[%v\"
	optimizer
J
0
1
2
3
$4
%5"
trackable_list_wrapper
J
0
1
2
3
$4
%5"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
1non_trainable_variables

2layers
3metrics
4layer_regularization_losses
5layer_metrics
	variables
trainable_variables
regularization_losses

__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
ξ2λ
(__inference_sequential_layer_call_fn_424
(__inference_sequential_layer_call_fn_612
(__inference_sequential_layer_call_fn_623
(__inference_sequential_layer_call_fn_571ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
Ϊ2Χ
C__inference_sequential_layer_call_and_return_conditional_losses_651
C__inference_sequential_layer_call_and_return_conditional_losses_679
C__inference_sequential_layer_call_and_return_conditional_losses_586
C__inference_sequential_layer_call_and_return_conditional_losses_601ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΙBΖ
__inference__wrapped_model_353input_1"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
6serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
­
7non_trainable_variables

8layers
9metrics
:layer_regularization_losses
;layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ο2Μ
%__inference_flatten_layer_call_fn_697’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_flatten_layer_call_and_return_conditional_losses_703’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
+:)
2sequential/dense/kernel
$:"2sequential/dense/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
<non_trainable_variables

=layers
>metrics
?layer_regularization_losses
@layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Ν2Κ
#__inference_dense_layer_call_fn_710’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
θ2ε
>__inference_dense_layer_call_and_return_conditional_losses_721’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
-:+
2sequential/dense_1/kernel
&:$2sequential/dense_1/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Anon_trainable_variables

Blayers
Cmetrics
Dlayer_regularization_losses
Elayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
Ο2Μ
%__inference_dense_1_layer_call_fn_728’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_dense_1_layer_call_and_return_conditional_losses_739’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,:*	
2sequential/dense_2/kernel
%:#
2sequential/dense_2/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
Fnon_trainable_variables

Glayers
Hmetrics
Ilayer_regularization_losses
Jlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
Ο2Μ
%__inference_dense_2_layer_call_fn_746’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
κ2η
@__inference_dense_2_layer_call_and_return_conditional_losses_757’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2training/Adam/iter
: (2training/Adam/beta_1
: (2training/Adam/beta_2
: (2training/Adam/decay
%:# (2training/Adam/learning_rate
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΘBΕ
!__inference_signature_wrapper_692input_1"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
^
	Ltotal
	Mcount
N
_fn_kwargs
O	variables
P	keras_api"
_tf_keras_metric
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
L0
M1"
trackable_list_wrapper
-
O	variables"
_generic_user_object
9:7
2'training/Adam/sequential/dense/kernel/m
2:02%training/Adam/sequential/dense/bias/m
;:9
2)training/Adam/sequential/dense_1/kernel/m
4:22'training/Adam/sequential/dense_1/bias/m
::8	
2)training/Adam/sequential/dense_2/kernel/m
3:1
2'training/Adam/sequential/dense_2/bias/m
9:7
2'training/Adam/sequential/dense/kernel/v
2:02%training/Adam/sequential/dense/bias/v
;:9
2)training/Adam/sequential/dense_1/kernel/v
4:22'training/Adam/sequential/dense_1/bias/v
::8	
2)training/Adam/sequential/dense_2/kernel/v
3:1
2'training/Adam/sequential/dense_2/bias/v
__inference__wrapped_model_353s$%4’1
*’'
%"
input_1?????????
ͺ "3ͺ0
.
output_1"
output_1?????????
’
@__inference_dense_1_layer_call_and_return_conditional_losses_739^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 z
%__inference_dense_1_layer_call_fn_728Q0’-
&’#
!
inputs?????????
ͺ "?????????‘
@__inference_dense_2_layer_call_and_return_conditional_losses_757]$%0’-
&’#
!
inputs?????????
ͺ "%’"

0?????????

 y
%__inference_dense_2_layer_call_fn_746P$%0’-
&’#
!
inputs?????????
ͺ "?????????
 
>__inference_dense_layer_call_and_return_conditional_losses_721^0’-
&’#
!
inputs?????????
ͺ "&’#

0?????????
 x
#__inference_dense_layer_call_fn_710Q0’-
&’#
!
inputs?????????
ͺ "?????????‘
@__inference_flatten_layer_call_and_return_conditional_losses_703]3’0
)’&
$!
inputs?????????
ͺ "&’#

0?????????
 y
%__inference_flatten_layer_call_fn_697P3’0
)’&
$!
inputs?????????
ͺ "?????????΄
C__inference_sequential_layer_call_and_return_conditional_losses_586m$%<’9
2’/
%"
input_1?????????
p 

 
ͺ "%’"

0?????????

 ΄
C__inference_sequential_layer_call_and_return_conditional_losses_601m$%<’9
2’/
%"
input_1?????????
p

 
ͺ "%’"

0?????????

 ³
C__inference_sequential_layer_call_and_return_conditional_losses_651l$%;’8
1’.
$!
inputs?????????
p 

 
ͺ "%’"

0?????????

 ³
C__inference_sequential_layer_call_and_return_conditional_losses_679l$%;’8
1’.
$!
inputs?????????
p

 
ͺ "%’"

0?????????

 
(__inference_sequential_layer_call_fn_424`$%<’9
2’/
%"
input_1?????????
p 

 
ͺ "?????????

(__inference_sequential_layer_call_fn_571`$%<’9
2’/
%"
input_1?????????
p

 
ͺ "?????????

(__inference_sequential_layer_call_fn_612_$%;’8
1’.
$!
inputs?????????
p 

 
ͺ "?????????

(__inference_sequential_layer_call_fn_623_$%;’8
1’.
$!
inputs?????????
p

 
ͺ "?????????
£
!__inference_signature_wrapper_692~$%?’<
’ 
5ͺ2
0
input_1%"
input_1?????????"3ͺ0
.
output_1"
output_1?????????
