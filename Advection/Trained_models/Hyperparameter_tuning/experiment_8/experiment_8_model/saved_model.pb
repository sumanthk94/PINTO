��
��
D
AddV2
x"T
y"T
z"T"
Ttype:
2	��
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( �
�
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
$
DisableCopyOnRead
resource�
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
�
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
.
Identity

input"T
output"T"	
Ttype
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	
�
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( �
?
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( ""
Ttype:
2	"
Tidxtype0:
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype�
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
d
Shape

input"T&
output"out_type��out_type"	
Ttype"
out_typetype0:
2	
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
�
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
executor_typestring ��
@
StaticRegexFullMatch	
input

output
"
patternstring
�
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
L

StringJoin
inputs*N

output"

Nint("
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
�
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.15.02v2.15.0-2-g0b15fdfcb3f8��
�
ou2/biasVarHandleOp*
_output_shapes
: *

debug_name	ou2/bias/*
dtype0*
shape:@*
shared_name
ou2/bias
a
ou2/bias/Read/ReadVariableOpReadVariableOpou2/bias*
_output_shapes
:@*
dtype0
�

ou2/kernelVarHandleOp*
_output_shapes
: *

debug_nameou2/kernel/*
dtype0*
shape
:@@*
shared_name
ou2/kernel
i
ou2/kernel/Read/ReadVariableOpReadVariableOp
ou2/kernel*
_output_shapes

:@@*
dtype0
�
ou1/biasVarHandleOp*
_output_shapes
: *

debug_name	ou1/bias/*
dtype0*
shape:@*
shared_name
ou1/bias
a
ou1/bias/Read/ReadVariableOpReadVariableOpou1/bias*
_output_shapes
:@*
dtype0
�

ou1/kernelVarHandleOp*
_output_shapes
: *

debug_nameou1/kernel/*
dtype0*
shape
:@@*
shared_name
ou1/kernel
i
ou1/kernel/Read/ReadVariableOpReadVariableOp
ou1/kernel*
_output_shapes

:@@*
dtype0
�
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *;

debug_name-+multi_head_attention/attention_output/bias/*
dtype0*
shape:@*;
shared_name,*multi_head_attention/attention_output/bias
�
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:@*
dtype0
�
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *=

debug_name/-multi_head_attention/attention_output/kernel/*
dtype0*
shape:@@*=
shared_name.,multi_head_attention/attention_output/kernel
�
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:@@*
dtype0
�
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/value/bias/*
dtype0*
shape
:@*0
shared_name!multi_head_attention/value/bias
�
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/value/kernel/*
dtype0*
shape:@@*2
shared_name#!multi_head_attention/value/kernel
�
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:@@*
dtype0
�
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *.

debug_name multi_head_attention/key/bias/*
dtype0*
shape
:@*.
shared_namemulti_head_attention/key/bias
�
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:@*
dtype0
�
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/key/kernel/*
dtype0*
shape:@@*0
shared_name!multi_head_attention/key/kernel
�
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:@@*
dtype0
�
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/query/bias/*
dtype0*
shape
:@*0
shared_name!multi_head_attention/query/bias
�
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:@*
dtype0
�
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/query/kernel/*
dtype0*
shape:@@*2
shared_name#!multi_head_attention/query/kernel
�
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:@@*
dtype0
�
bpe_layer2/biasVarHandleOp*
_output_shapes
: * 

debug_namebpe_layer2/bias/*
dtype0*
shape:@* 
shared_namebpe_layer2/bias
o
#bpe_layer2/bias/Read/ReadVariableOpReadVariableOpbpe_layer2/bias*
_output_shapes
:@*
dtype0
�
bpe_layer2/kernelVarHandleOp*
_output_shapes
: *"

debug_namebpe_layer2/kernel/*
dtype0*
shape
:@@*"
shared_namebpe_layer2/kernel
w
%bpe_layer2/kernel/Read/ReadVariableOpReadVariableOpbpe_layer2/kernel*
_output_shapes

:@@*
dtype0
�
bpe_layer1/biasVarHandleOp*
_output_shapes
: * 

debug_namebpe_layer1/bias/*
dtype0*
shape:@* 
shared_namebpe_layer1/bias
o
#bpe_layer1/bias/Read/ReadVariableOpReadVariableOpbpe_layer1/bias*
_output_shapes
:@*
dtype0
�
bpe_layer1/kernelVarHandleOp*
_output_shapes
: *"

debug_namebpe_layer1/kernel/*
dtype0*
shape
:@*"
shared_namebpe_layer1/kernel
w
%bpe_layer1/kernel/Read/ReadVariableOpReadVariableOpbpe_layer1/kernel*
_output_shapes

:@*
dtype0
�
spatial_layer2/biasVarHandleOp*
_output_shapes
: *$

debug_namespatial_layer2/bias/*
dtype0*
shape:@*$
shared_namespatial_layer2/bias
w
'spatial_layer2/bias/Read/ReadVariableOpReadVariableOpspatial_layer2/bias*
_output_shapes
:@*
dtype0
�
spatial_layer2/kernelVarHandleOp*
_output_shapes
: *&

debug_namespatial_layer2/kernel/*
dtype0*
shape
:@@*&
shared_namespatial_layer2/kernel

)spatial_layer2/kernel/Read/ReadVariableOpReadVariableOpspatial_layer2/kernel*
_output_shapes

:@@*
dtype0
�
spatial_layer1/biasVarHandleOp*
_output_shapes
: *$

debug_namespatial_layer1/bias/*
dtype0*
shape:@*$
shared_namespatial_layer1/bias
w
'spatial_layer1/bias/Read/ReadVariableOpReadVariableOpspatial_layer1/bias*
_output_shapes
:@*
dtype0
�
spatial_layer1/kernelVarHandleOp*
_output_shapes
: *&

debug_namespatial_layer1/kernel/*
dtype0*
shape
:@*&
shared_namespatial_layer1/kernel

)spatial_layer1/kernel/Read/ReadVariableOpReadVariableOpspatial_layer1/kernel*
_output_shapes

:@*
dtype0
�
output_u/kernelVarHandleOp*
_output_shapes
: * 

debug_nameoutput_u/kernel/*
dtype0*
shape
:@* 
shared_nameoutput_u/kernel
s
#output_u/kernel/Read/ReadVariableOpReadVariableOpoutput_u/kernel*
_output_shapes

:@*
dtype0
�

dense/biasVarHandleOp*
_output_shapes
: *

debug_namedense/bias/*
dtype0*
shape:@*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:@*
dtype0
�
dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
:@@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@@*
dtype0
�
bve_layer_2/biasVarHandleOp*
_output_shapes
: *!

debug_namebve_layer_2/bias/*
dtype0*
shape:@*!
shared_namebve_layer_2/bias
q
$bve_layer_2/bias/Read/ReadVariableOpReadVariableOpbve_layer_2/bias*
_output_shapes
:@*
dtype0
�
bve_layer_2/kernelVarHandleOp*
_output_shapes
: *#

debug_namebve_layer_2/kernel/*
dtype0*
shape
:@@*#
shared_namebve_layer_2/kernel
y
&bve_layer_2/kernel/Read/ReadVariableOpReadVariableOpbve_layer_2/kernel*
_output_shapes

:@@*
dtype0
�
bve_layer_1/biasVarHandleOp*
_output_shapes
: *!

debug_namebve_layer_1/bias/*
dtype0*
shape:@*!
shared_namebve_layer_1/bias
q
$bve_layer_1/bias/Read/ReadVariableOpReadVariableOpbve_layer_1/bias*
_output_shapes
:@*
dtype0
�
bve_layer_1/kernelVarHandleOp*
_output_shapes
: *#

debug_namebve_layer_1/kernel/*
dtype0*
shape
:@*#
shared_namebve_layer_1/kernel
y
&bve_layer_1/kernel/Read/ReadVariableOpReadVariableOpbve_layer_1/kernel*
_output_shapes

:@*
dtype0
�
serving_default_Xbc_layerPlaceholder*4
_output_shapes"
 :������������������*
dtype0*)
shape :������������������
z
serving_default_t_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
serving_default_tbc_layerPlaceholder*4
_output_shapes"
 :������������������*
dtype0*)
shape :������������������
�
serving_default_ubc_layerPlaceholder*4
_output_shapes"
 :������������������*
dtype0*)
shape :������������������
z
serving_default_x_inputPlaceholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_Xbc_layerserving_default_t_inputserving_default_tbc_layerserving_default_ubc_layerserving_default_x_inputbve_layer_1/kernelbve_layer_1/biasspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/biasbpe_layer1/kernelbpe_layer1/biasbpe_layer2/kernelbpe_layer2/biasbve_layer_2/kernelbve_layer_2/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biasdense/kernel
dense/bias
ou1/kernelou1/bias
ou2/kernelou2/biasoutput_u/kernel*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� */
f*R(
&__inference_signature_wrapper_44969333

NoOpNoOp
ۼ
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*��
value��B�� B��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer_with_weights-2
layer-14
layer_with_weights-3
layer-15
layer_with_weights-4
layer-16
layer-17
layer_with_weights-5
layer-18
layer-19
layer_with_weights-6
layer-20
layer_with_weights-7
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures*
* 
* 
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses* 
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses* 
* 
* 
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses* 
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses* 
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses* 
* 
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses* 
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses* 
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias*
�
Qlayer_with_weights-0
Qlayer-0
Rlayer-1
Slayer_with_weights-1
Slayer-2
Tlayer-3
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses*
�
[layer_with_weights-0
[layer-0
\layer-1
]layer_with_weights-1
]layer-2
^layer-3
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses*
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias*
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_query_dense
t
_key_dense
u_value_dense
v_softmax
w_dropout_layer
x_output_dense*
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses* 
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�layer_with_weights-0
�layer-0
�layer-1
�layer_with_weights-1
�layer-2
�layer-3
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel*
�
O0
P1
�2
�3
�4
�5
�6
�7
�8
�9
k10
l11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26*
�
O0
P1
�2
�3
�4
�5
�6
�7
�8
�9
k10
l11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
* 

�serving_default* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

O0
P1*

O0
P1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEbve_layer_1/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbve_layer_1/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

k0
l1*

k0
l1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
b\
VARIABLE_VALUEbve_layer_2/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEbve_layer_2/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*
D
�0
�1
�2
�3
�4
�5
�6
�7*
D
�0
�1
�2
�3
�4
�5
�6
�7*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias*
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses* 
$
�0
�1
�2
�3*
$
�0
�1
�2
�3*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0
�trace_1* 

�trace_0
�trace_1* 

�0*

�0*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
_Y
VARIABLE_VALUEoutput_u/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEspatial_layer1/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEspatial_layer1/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEspatial_layer2/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEspatial_layer2/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEbpe_layer1/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEbpe_layer1/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEbpe_layer2/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEbpe_layer2/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention/query/kernel'variables/12/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/query/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/key/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEmulti_head_attention/key/bias'variables/15/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention/value/kernel'variables/16/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention/value/bias'variables/17/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention/attention_output/kernel'variables/18/.ATTRIBUTES/VARIABLE_VALUE*
ke
VARIABLE_VALUE*multi_head_attention/attention_output/bias'variables/19/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
ou1/kernel'variables/22/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEou1/bias'variables/23/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
ou2/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEou2/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
* 
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21*
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

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
 
Q0
R1
S2
T3*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
 
[0
\1
]2
^3*
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
.
s0
t1
u2
v3
w4
x5*
* 
* 
* 
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 
* 
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*
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
* 
* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 

�0
�1*

�0
�1*
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses*

�trace_0* 

�trace_0* 
* 
* 
* 
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses* 

�trace_0* 

�trace_0* 
* 
$
�0
�1
�2
�3*
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
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamebve_layer_1/kernelbve_layer_1/biasbve_layer_2/kernelbve_layer_2/biasdense/kernel
dense/biasoutput_u/kernelspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/biasbpe_layer1/kernelbpe_layer1/biasbpe_layer2/kernelbpe_layer2/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias
ou1/kernelou1/bias
ou2/kernelou2/biasConst*(
Tin!
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� **
f%R#
!__inference__traced_save_44970147
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamebve_layer_1/kernelbve_layer_1/biasbve_layer_2/kernelbve_layer_2/biasdense/kernel
dense/biasoutput_u/kernelspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/biasbpe_layer1/kernelbpe_layer1/biasbpe_layer2/kernelbpe_layer2/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias
ou1/kernelou1/bias
ou2/kernelou2/bias*'
Tin 
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *-
f(R&
$__inference__traced_restore_44970237��
�
�
H__inference_bpe_layer2_layer_call_and_return_conditional_losses_44969891

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :������������������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
F
*__inference_flatten_layer_call_fn_44969685

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_44968878`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
(__inference_model_layer_call_fn_44969141
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layer
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallx_inputt_input	xbc_layer	tbc_layer	ubc_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_44969015o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:������������������:������������������:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969137:($
"
_user_specified_name
44969135:($
"
_user_specified_name
44969133:($
"
_user_specified_name
44969131:($
"
_user_specified_name
44969129:($
"
_user_specified_name
44969127:($
"
_user_specified_name
44969125:($
"
_user_specified_name
44969123:($
"
_user_specified_name
44969121:($
"
_user_specified_name
44969119:($
"
_user_specified_name
44969117:($
"
_user_specified_name
44969115:($
"
_user_specified_name
44969113:($
"
_user_specified_name
44969111:($
"
_user_specified_name
44969109:($
"
_user_specified_name
44969107:($
"
_user_specified_name
44969105:($
"
_user_specified_name
44969103:($
"
_user_specified_name
44969101:($
"
_user_specified_name
44969099:($
"
_user_specified_name
44969097:(
$
"
_user_specified_name
44969095:(	$
"
_user_specified_name
44969093:($
"
_user_specified_name
44969091:($
"
_user_specified_name
44969089:($
"
_user_specified_name
44969087:($
"
_user_specified_name
44969085:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:���������
!
_user_specified_name	t_input:P L
'
_output_shapes
:���������
!
_user_specified_name	x_input
�
w
K__inference_concatenate_1_layer_call_and_return_conditional_losses_44969430
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������d
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������:������������������:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
�
I__inference_bve_layer_2_layer_call_and_return_conditional_losses_44969510

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :������������������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@]
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@d
IdentityIdentityTanh:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
f
J__inference_activation_4_layer_call_and_return_conditional_losses_44968519

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:���������@P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_bpe_layer1_layer_call_and_return_conditional_losses_44968373

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
��
�
!__inference__traced_save_44970147
file_prefix;
)read_disablecopyonread_bve_layer_1_kernel:@7
)read_1_disablecopyonread_bve_layer_1_bias:@=
+read_2_disablecopyonread_bve_layer_2_kernel:@@7
)read_3_disablecopyonread_bve_layer_2_bias:@7
%read_4_disablecopyonread_dense_kernel:@@1
#read_5_disablecopyonread_dense_bias:@:
(read_6_disablecopyonread_output_u_kernel:@@
.read_7_disablecopyonread_spatial_layer1_kernel:@:
,read_8_disablecopyonread_spatial_layer1_bias:@@
.read_9_disablecopyonread_spatial_layer2_kernel:@@;
-read_10_disablecopyonread_spatial_layer2_bias:@=
+read_11_disablecopyonread_bpe_layer1_kernel:@7
)read_12_disablecopyonread_bpe_layer1_bias:@=
+read_13_disablecopyonread_bpe_layer2_kernel:@@7
)read_14_disablecopyonread_bpe_layer2_bias:@Q
;read_15_disablecopyonread_multi_head_attention_query_kernel:@@K
9read_16_disablecopyonread_multi_head_attention_query_bias:@O
9read_17_disablecopyonread_multi_head_attention_key_kernel:@@I
7read_18_disablecopyonread_multi_head_attention_key_bias:@Q
;read_19_disablecopyonread_multi_head_attention_value_kernel:@@K
9read_20_disablecopyonread_multi_head_attention_value_bias:@\
Fread_21_disablecopyonread_multi_head_attention_attention_output_kernel:@@R
Dread_22_disablecopyonread_multi_head_attention_attention_output_bias:@6
$read_23_disablecopyonread_ou1_kernel:@@0
"read_24_disablecopyonread_ou1_bias:@6
$read_25_disablecopyonread_ou2_kernel:@@0
"read_26_disablecopyonread_ou2_bias:@
savev2_const
identity_55��MergeV2Checkpoints�Read/DisableCopyOnRead�Read/ReadVariableOp�Read_1/DisableCopyOnRead�Read_1/ReadVariableOp�Read_10/DisableCopyOnRead�Read_10/ReadVariableOp�Read_11/DisableCopyOnRead�Read_11/ReadVariableOp�Read_12/DisableCopyOnRead�Read_12/ReadVariableOp�Read_13/DisableCopyOnRead�Read_13/ReadVariableOp�Read_14/DisableCopyOnRead�Read_14/ReadVariableOp�Read_15/DisableCopyOnRead�Read_15/ReadVariableOp�Read_16/DisableCopyOnRead�Read_16/ReadVariableOp�Read_17/DisableCopyOnRead�Read_17/ReadVariableOp�Read_18/DisableCopyOnRead�Read_18/ReadVariableOp�Read_19/DisableCopyOnRead�Read_19/ReadVariableOp�Read_2/DisableCopyOnRead�Read_2/ReadVariableOp�Read_20/DisableCopyOnRead�Read_20/ReadVariableOp�Read_21/DisableCopyOnRead�Read_21/ReadVariableOp�Read_22/DisableCopyOnRead�Read_22/ReadVariableOp�Read_23/DisableCopyOnRead�Read_23/ReadVariableOp�Read_24/DisableCopyOnRead�Read_24/ReadVariableOp�Read_25/DisableCopyOnRead�Read_25/ReadVariableOp�Read_26/DisableCopyOnRead�Read_26/ReadVariableOp�Read_3/DisableCopyOnRead�Read_3/ReadVariableOp�Read_4/DisableCopyOnRead�Read_4/ReadVariableOp�Read_5/DisableCopyOnRead�Read_5/ReadVariableOp�Read_6/DisableCopyOnRead�Read_6/ReadVariableOp�Read_7/DisableCopyOnRead�Read_7/ReadVariableOp�Read_8/DisableCopyOnRead�Read_8/ReadVariableOp�Read_9/DisableCopyOnRead�Read_9/ReadVariableOpw
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
_temp/part�
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
value	B : �
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: {
Read/DisableCopyOnReadDisableCopyOnRead)read_disablecopyonread_bve_layer_1_kernel"/device:CPU:0*
_output_shapes
 �
Read/ReadVariableOpReadVariableOp)read_disablecopyonread_bve_layer_1_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:@}
Read_1/DisableCopyOnReadDisableCopyOnRead)read_1_disablecopyonread_bve_layer_1_bias"/device:CPU:0*
_output_shapes
 �
Read_1/ReadVariableOpReadVariableOp)read_1_disablecopyonread_bve_layer_1_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_2IdentityRead_1/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_3IdentityIdentity_2:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_2/DisableCopyOnReadDisableCopyOnRead+read_2_disablecopyonread_bve_layer_2_kernel"/device:CPU:0*
_output_shapes
 �
Read_2/ReadVariableOpReadVariableOp+read_2_disablecopyonread_bve_layer_2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@@}
Read_3/DisableCopyOnReadDisableCopyOnRead)read_3_disablecopyonread_bve_layer_2_bias"/device:CPU:0*
_output_shapes
 �
Read_3/ReadVariableOpReadVariableOp)read_3_disablecopyonread_bve_layer_2_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0i

Identity_6IdentityRead_3/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@_

Identity_7IdentityIdentity_6:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_4/DisableCopyOnReadDisableCopyOnRead%read_4_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 �
Read_4/ReadVariableOpReadVariableOp%read_4_disablecopyonread_dense_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0m

Identity_8IdentityRead_4/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@c

Identity_9IdentityIdentity_8:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_5/DisableCopyOnReadDisableCopyOnRead#read_5_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 �
Read_5/ReadVariableOpReadVariableOp#read_5_disablecopyonread_dense_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_10IdentityRead_5/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_11IdentityIdentity_10:output:0"/device:CPU:0*
T0*
_output_shapes
:@|
Read_6/DisableCopyOnReadDisableCopyOnRead(read_6_disablecopyonread_output_u_kernel"/device:CPU:0*
_output_shapes
 �
Read_6/ReadVariableOpReadVariableOp(read_6_disablecopyonread_output_u_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_7/DisableCopyOnReadDisableCopyOnRead.read_7_disablecopyonread_spatial_layer1_kernel"/device:CPU:0*
_output_shapes
 �
Read_7/ReadVariableOpReadVariableOp.read_7_disablecopyonread_spatial_layer1_kernel^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0n
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_8/DisableCopyOnReadDisableCopyOnRead,read_8_disablecopyonread_spatial_layer1_bias"/device:CPU:0*
_output_shapes
 �
Read_8/ReadVariableOpReadVariableOp,read_8_disablecopyonread_spatial_layer1_bias^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_9/DisableCopyOnReadDisableCopyOnRead.read_9_disablecopyonread_spatial_layer2_kernel"/device:CPU:0*
_output_shapes
 �
Read_9/ReadVariableOpReadVariableOp.read_9_disablecopyonread_spatial_layer2_kernel^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0n
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes

:@@�
Read_10/DisableCopyOnReadDisableCopyOnRead-read_10_disablecopyonread_spatial_layer2_bias"/device:CPU:0*
_output_shapes
 �
Read_10/ReadVariableOpReadVariableOp-read_10_disablecopyonread_spatial_layer2_bias^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_11/DisableCopyOnReadDisableCopyOnRead+read_11_disablecopyonread_bpe_layer1_kernel"/device:CPU:0*
_output_shapes
 �
Read_11/ReadVariableOpReadVariableOp+read_11_disablecopyonread_bpe_layer1_kernel^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes

:@~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_bpe_layer1_bias"/device:CPU:0*
_output_shapes
 �
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_bpe_layer1_bias^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_13/DisableCopyOnReadDisableCopyOnRead+read_13_disablecopyonread_bpe_layer2_kernel"/device:CPU:0*
_output_shapes
 �
Read_13/ReadVariableOpReadVariableOp+read_13_disablecopyonread_bpe_layer2_kernel^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes

:@@~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_bpe_layer2_bias"/device:CPU:0*
_output_shapes
 �
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_bpe_layer2_bias^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes
:@�
Read_15/DisableCopyOnReadDisableCopyOnRead;read_15_disablecopyonread_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 �
Read_15/ReadVariableOpReadVariableOp;read_15_disablecopyonread_multi_head_attention_query_kernel^Read_15/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@�
Read_16/DisableCopyOnReadDisableCopyOnRead9read_16_disablecopyonread_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 �
Read_16/ReadVariableOpReadVariableOp9read_16_disablecopyonread_multi_head_attention_query_bias^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_17/DisableCopyOnReadDisableCopyOnRead9read_17_disablecopyonread_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 �
Read_17/ReadVariableOpReadVariableOp9read_17_disablecopyonread_multi_head_attention_key_kernel^Read_17/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@�
Read_18/DisableCopyOnReadDisableCopyOnRead7read_18_disablecopyonread_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 �
Read_18/ReadVariableOpReadVariableOp7read_18_disablecopyonread_multi_head_attention_key_bias^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_19/DisableCopyOnReadDisableCopyOnRead;read_19_disablecopyonread_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 �
Read_19/ReadVariableOpReadVariableOp;read_19_disablecopyonread_multi_head_attention_value_kernel^Read_19/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@�
Read_20/DisableCopyOnReadDisableCopyOnRead9read_20_disablecopyonread_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 �
Read_20/ReadVariableOpReadVariableOp9read_20_disablecopyonread_multi_head_attention_value_bias^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:@�
Read_21/DisableCopyOnReadDisableCopyOnReadFread_21_disablecopyonread_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 �
Read_21/ReadVariableOpReadVariableOpFread_21_disablecopyonread_multi_head_attention_attention_output_kernel^Read_21/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@�
Read_22/DisableCopyOnReadDisableCopyOnReadDread_22_disablecopyonread_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 �
Read_22/ReadVariableOpReadVariableOpDread_22_disablecopyonread_multi_head_attention_attention_output_bias^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_23/DisableCopyOnReadDisableCopyOnRead$read_23_disablecopyonread_ou1_kernel"/device:CPU:0*
_output_shapes
 �
Read_23/ReadVariableOpReadVariableOp$read_23_disablecopyonread_ou1_kernel^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_24/DisableCopyOnReadDisableCopyOnRead"read_24_disablecopyonread_ou1_bias"/device:CPU:0*
_output_shapes
 �
Read_24/ReadVariableOpReadVariableOp"read_24_disablecopyonread_ou1_bias^Read_24/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_25/DisableCopyOnReadDisableCopyOnRead$read_25_disablecopyonread_ou2_kernel"/device:CPU:0*
_output_shapes
 �
Read_25/ReadVariableOpReadVariableOp$read_25_disablecopyonread_ou2_kernel^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_26/DisableCopyOnReadDisableCopyOnRead"read_26_disablecopyonread_ou2_bias"/device:CPU:0*
_output_shapes
 �
Read_26/ReadVariableOpReadVariableOp"read_26_disablecopyonread_ou2_bias^Read_26/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*
_output_shapes
:@�

SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 **
dtypes 
2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_54Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_55IdentityIdentity_54:output:0^NoOp*
T0*
_output_shapes
: �
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_55Identity_55:output:0*(
_construction_contextkEagerRuntime*M
_input_shapes<
:: : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
MergeV2CheckpointsMergeV2Checkpoints20
Read/DisableCopyOnReadRead/DisableCopyOnRead2*
Read/ReadVariableOpRead/ReadVariableOp24
Read_1/DisableCopyOnReadRead_1/DisableCopyOnRead2.
Read_1/ReadVariableOpRead_1/ReadVariableOp26
Read_10/DisableCopyOnReadRead_10/DisableCopyOnRead20
Read_10/ReadVariableOpRead_10/ReadVariableOp26
Read_11/DisableCopyOnReadRead_11/DisableCopyOnRead20
Read_11/ReadVariableOpRead_11/ReadVariableOp26
Read_12/DisableCopyOnReadRead_12/DisableCopyOnRead20
Read_12/ReadVariableOpRead_12/ReadVariableOp26
Read_13/DisableCopyOnReadRead_13/DisableCopyOnRead20
Read_13/ReadVariableOpRead_13/ReadVariableOp26
Read_14/DisableCopyOnReadRead_14/DisableCopyOnRead20
Read_14/ReadVariableOpRead_14/ReadVariableOp26
Read_15/DisableCopyOnReadRead_15/DisableCopyOnRead20
Read_15/ReadVariableOpRead_15/ReadVariableOp26
Read_16/DisableCopyOnReadRead_16/DisableCopyOnRead20
Read_16/ReadVariableOpRead_16/ReadVariableOp26
Read_17/DisableCopyOnReadRead_17/DisableCopyOnRead20
Read_17/ReadVariableOpRead_17/ReadVariableOp26
Read_18/DisableCopyOnReadRead_18/DisableCopyOnRead20
Read_18/ReadVariableOpRead_18/ReadVariableOp26
Read_19/DisableCopyOnReadRead_19/DisableCopyOnRead20
Read_19/ReadVariableOpRead_19/ReadVariableOp24
Read_2/DisableCopyOnReadRead_2/DisableCopyOnRead2.
Read_2/ReadVariableOpRead_2/ReadVariableOp26
Read_20/DisableCopyOnReadRead_20/DisableCopyOnRead20
Read_20/ReadVariableOpRead_20/ReadVariableOp26
Read_21/DisableCopyOnReadRead_21/DisableCopyOnRead20
Read_21/ReadVariableOpRead_21/ReadVariableOp26
Read_22/DisableCopyOnReadRead_22/DisableCopyOnRead20
Read_22/ReadVariableOpRead_22/ReadVariableOp26
Read_23/DisableCopyOnReadRead_23/DisableCopyOnRead20
Read_23/ReadVariableOpRead_23/ReadVariableOp26
Read_24/DisableCopyOnReadRead_24/DisableCopyOnRead20
Read_24/ReadVariableOpRead_24/ReadVariableOp26
Read_25/DisableCopyOnReadRead_25/DisableCopyOnRead20
Read_25/ReadVariableOpRead_25/ReadVariableOp26
Read_26/DisableCopyOnReadRead_26/DisableCopyOnRead20
Read_26/ReadVariableOpRead_26/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp24
Read_4/DisableCopyOnReadRead_4/DisableCopyOnRead2.
Read_4/ReadVariableOpRead_4/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=9

_output_shapes
: 

_user_specified_nameConst:($
"
_user_specified_name
ou2/bias:*&
$
_user_specified_name
ou2/kernel:($
"
_user_specified_name
ou1/bias:*&
$
_user_specified_name
ou1/kernel:JF
D
_user_specified_name,*multi_head_attention/attention_output/bias:LH
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?;
9
_user_specified_name!multi_head_attention/value/bias:A=
;
_user_specified_name#!multi_head_attention/value/kernel:=9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:/+
)
_user_specified_namebpe_layer2/bias:1-
+
_user_specified_namebpe_layer2/kernel:/+
)
_user_specified_namebpe_layer1/bias:1-
+
_user_specified_namebpe_layer1/kernel:3/
-
_user_specified_namespatial_layer2/bias:5
1
/
_user_specified_namespatial_layer2/kernel:3	/
-
_user_specified_namespatial_layer1/bias:51
/
_user_specified_namespatial_layer1/kernel:/+
)
_user_specified_nameoutput_u/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:0,
*
_user_specified_namebve_layer_2/bias:2.
,
_user_specified_namebve_layer_2/kernel:0,
*
_user_specified_namebve_layer_1/bias:2.
,
_user_specified_namebve_layer_1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
9__inference_spatial_transformation_layer_call_fn_44968313
spatial_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968287s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44968309:($
"
_user_specified_name
44968307:($
"
_user_specified_name
44968305:($
"
_user_specified_name
44968303:a ]
+
_output_shapes
:���������
.
_user_specified_namespatial_layer1_input
�
�
-__inference_bpe_layer1_layer_call_fn_44969812

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_bpe_layer1_layer_call_and_return_conditional_losses_44968373|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969808:($
"
_user_specified_name
44969806:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�Y
�
C__inference_model_layer_call_and_return_conditional_losses_44969015
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layer&
bve_layer_1_44968912:@"
bve_layer_1_44968914:@1
spatial_transformation_44968919:@-
spatial_transformation_44968921:@1
spatial_transformation_44968923:@@-
spatial_transformation_44968925:@
bpe_44968928:@
bpe_44968930:@
bpe_44968932:@@
bpe_44968934:@&
bve_layer_2_44968937:@@"
bve_layer_2_44968939:@3
multi_head_attention_44968978:@@/
multi_head_attention_44968980:@3
multi_head_attention_44968982:@@/
multi_head_attention_44968984:@3
multi_head_attention_44968986:@@/
multi_head_attention_44968988:@3
multi_head_attention_44968990:@@+
multi_head_attention_44968992:@ 
dense_44968996:@@
dense_44968998:@

u_44969002:@@

u_44969004:@

u_44969006:@@

u_44969008:@#
output_u_44969011:@
identity��BPE/StatefulPartitionedCall�U/StatefulPartitionedCall�#bve_layer_1/StatefulPartitionedCall�#bve_layer_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall� output_u/StatefulPartitionedCall�.spatial_transformation/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallx_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_rescaling_layer_call_and_return_conditional_losses_44968627�
rescaling_1/PartitionedCallPartitionedCallt_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_1_layer_call_and_return_conditional_losses_44968636�
rescaling_2/PartitionedCallPartitionedCall	xbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_2_layer_call_and_return_conditional_losses_44968646�
rescaling_3/PartitionedCallPartitionedCall	tbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_3_layer_call_and_return_conditional_losses_44968655�
concatenate/PartitionedCallPartitionedCall"rescaling/PartitionedCall:output:0$rescaling_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_44968663�
#bve_layer_1/StatefulPartitionedCallStatefulPartitionedCall	ubc_layerbve_layer_1_44968912bve_layer_1_44968914*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_bve_layer_1_layer_call_and_return_conditional_losses_44968695�
concatenate_1/PartitionedCallPartitionedCall$rescaling_2/PartitionedCall:output:0$rescaling_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_44968707�
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_44968721�
.spatial_transformation/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0spatial_transformation_44968919spatial_transformation_44968921spatial_transformation_44968923spatial_transformation_44968925*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968287�
BPE/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0bpe_44968928bpe_44968930bpe_44968932bpe_44968934*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_BPE_layer_call_and_return_conditional_losses_44968443�
#bve_layer_2/StatefulPartitionedCallStatefulPartitionedCall,bve_layer_1/StatefulPartitionedCall:output:0bve_layer_2_44968937bve_layer_2_44968939*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_bve_layer_2_layer_call_and_return_conditional_losses_44968771�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:0,bve_layer_2/StatefulPartitionedCall:output:0$BPE/StatefulPartitionedCall:output:0multi_head_attention_44968978multi_head_attention_44968980multi_head_attention_44968982multi_head_attention_44968984multi_head_attention_44968986multi_head_attention_44968988multi_head_attention_44968990multi_head_attention_44968992*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44968977�
add/PartitionedCallPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:05multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_44968835�
dense/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0dense_44968996dense_44968998*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_44968867�
flatten/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_44968878�
U/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
u_44969002
u_44969004
u_44969006
u_44969008*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_U_layer_call_and_return_conditional_losses_44968559�
 output_u/StatefulPartitionedCallStatefulPartitionedCall"U/StatefulPartitionedCall:output:0output_u_44969011*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_output_u_layer_call_and_return_conditional_losses_44968895x
IdentityIdentity)output_u/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BPE/StatefulPartitionedCall^U/StatefulPartitionedCall$^bve_layer_1/StatefulPartitionedCall$^bve_layer_2/StatefulPartitionedCall^dense/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall!^output_u/StatefulPartitionedCall/^spatial_transformation/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:������������������:������������������:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
BPE/StatefulPartitionedCallBPE/StatefulPartitionedCall26
U/StatefulPartitionedCallU/StatefulPartitionedCall2J
#bve_layer_1/StatefulPartitionedCall#bve_layer_1/StatefulPartitionedCall2J
#bve_layer_2/StatefulPartitionedCall#bve_layer_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2D
 output_u/StatefulPartitionedCall output_u/StatefulPartitionedCall2`
.spatial_transformation/StatefulPartitionedCall.spatial_transformation/StatefulPartitionedCall:($
"
_user_specified_name
44969011:($
"
_user_specified_name
44969008:($
"
_user_specified_name
44969006:($
"
_user_specified_name
44969004:($
"
_user_specified_name
44969002:($
"
_user_specified_name
44968998:($
"
_user_specified_name
44968996:($
"
_user_specified_name
44968992:($
"
_user_specified_name
44968990:($
"
_user_specified_name
44968988:($
"
_user_specified_name
44968986:($
"
_user_specified_name
44968984:($
"
_user_specified_name
44968982:($
"
_user_specified_name
44968980:($
"
_user_specified_name
44968978:($
"
_user_specified_name
44968939:($
"
_user_specified_name
44968937:($
"
_user_specified_name
44968934:($
"
_user_specified_name
44968932:($
"
_user_specified_name
44968930:($
"
_user_specified_name
44968928:(
$
"
_user_specified_name
44968925:(	$
"
_user_specified_name
44968923:($
"
_user_specified_name
44968921:($
"
_user_specified_name
44968919:($
"
_user_specified_name
44968914:($
"
_user_specified_name
44968912:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:���������
!
_user_specified_name	t_input:P L
'
_output_shapes
:���������
!
_user_specified_name	x_input
�.
�
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44969592	
query	
value
keyA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"������������������*
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"������������������z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"�������������������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������@*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������@:������������������@:������������������@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:YU
4
_output_shapes"
 :������������������@

_user_specified_namekey:[W
4
_output_shapes"
 :������������������@

_user_specified_namevalue:R N
+
_output_shapes
:���������@

_user_specified_namequery
�
�
C__inference_dense_layer_call_and_return_conditional_losses_44969680

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:���������@[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:���������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
\
0__inference_concatenate_1_layer_call_fn_44969423
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_44968707m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������:������������������:^Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :������������������
"
_user_specified_name
inputs_0
�
e
I__inference_rescaling_1_layer_call_and_return_conditional_losses_44968636

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ��U
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:���������Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
A__inference_ou2_layer_call_and_return_conditional_losses_44969949

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
H__inference_bpe_layer2_layer_call_and_return_conditional_losses_44968414

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :������������������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�.
�
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44969628	
query	
value
keyA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"������������������*
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"������������������z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"�������������������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������@*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������@:������������������@:������������������@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:YU
4
_output_shapes"
 :������������������@

_user_specified_namekey:[W
4
_output_shapes"
 :������������������@

_user_specified_namevalue:R N
+
_output_shapes
:���������@

_user_specified_namequery
�
k
A__inference_add_layer_call_and_return_conditional_losses_44968835

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:���������@S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������@:���������@:SO
+
_output_shapes
:���������@
 
_user_specified_nameinputs:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_ou2_layer_call_fn_44969939

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_ou2_layer_call_and_return_conditional_losses_44968530o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969935:($
"
_user_specified_name
44969933:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
&__inference_signature_wrapper_44969333
	xbc_layer
t_input
	tbc_layer
	ubc_layer
x_input
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallx_inputt_input	xbc_layer	tbc_layer	ubc_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *,
f'R%
#__inference__wrapped_model_44968185o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:������������������:���������:������������������:������������������:���������: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969329:($
"
_user_specified_name
44969327:($
"
_user_specified_name
44969325:($
"
_user_specified_name
44969323:($
"
_user_specified_name
44969321:($
"
_user_specified_name
44969319:($
"
_user_specified_name
44969317:($
"
_user_specified_name
44969315:($
"
_user_specified_name
44969313:($
"
_user_specified_name
44969311:($
"
_user_specified_name
44969309:($
"
_user_specified_name
44969307:($
"
_user_specified_name
44969305:($
"
_user_specified_name
44969303:($
"
_user_specified_name
44969301:($
"
_user_specified_name
44969299:($
"
_user_specified_name
44969297:($
"
_user_specified_name
44969295:($
"
_user_specified_name
44969293:($
"
_user_specified_name
44969291:($
"
_user_specified_name
44969289:(
$
"
_user_specified_name
44969287:(	$
"
_user_specified_name
44969285:($
"
_user_specified_name
44969283:($
"
_user_specified_name
44969281:($
"
_user_specified_name
44969279:($
"
_user_specified_name
44969277:PL
'
_output_shapes
:���������
!
_user_specified_name	x_input:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	tbc_layer:PL
'
_output_shapes
:���������
!
_user_specified_name	t_input:_ [
4
_output_shapes"
 :������������������
#
_user_specified_name	Xbc_layer
�
�
?__inference_U_layer_call_and_return_conditional_losses_44968543
	ou1_input
ou1_44968510:@@
ou1_44968512:@
ou2_44968531:@@
ou2_44968533:@
identity��ou1/StatefulPartitionedCall�ou2/StatefulPartitionedCall�
ou1/StatefulPartitionedCallStatefulPartitionedCall	ou1_inputou1_44968510ou1_44968512*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_ou1_layer_call_and_return_conditional_losses_44968509�
activation_4/PartitionedCallPartitionedCall$ou1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_44968519�
ou2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0ou2_44968531ou2_44968533*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_ou2_layer_call_and_return_conditional_losses_44968530�
activation_5/PartitionedCallPartitionedCall$ou2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_44968540t
IdentityIdentity%activation_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@^
NoOpNoOp^ou1/StatefulPartitionedCall^ou2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2:
ou1/StatefulPartitionedCallou1/StatefulPartitionedCall2:
ou2/StatefulPartitionedCallou2/StatefulPartitionedCall:($
"
_user_specified_name
44968533:($
"
_user_specified_name
44968531:($
"
_user_specified_name
44968512:($
"
_user_specified_name
44968510:R N
'
_output_shapes
:���������@
#
_user_specified_name	ou1_input
�
�
(__inference_dense_layer_call_fn_44969649

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_44968867s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969645:($
"
_user_specified_name
44969643:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
A__inference_BPE_layer_call_and_return_conditional_losses_44968427
bpe_layer1_input%
bpe_layer1_44968374:@!
bpe_layer1_44968376:@%
bpe_layer2_44968415:@@!
bpe_layer2_44968417:@
identity��"bpe_layer1/StatefulPartitionedCall�"bpe_layer2/StatefulPartitionedCall�
"bpe_layer1/StatefulPartitionedCallStatefulPartitionedCallbpe_layer1_inputbpe_layer1_44968374bpe_layer1_44968376*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_bpe_layer1_layer_call_and_return_conditional_losses_44968373�
activation_2/PartitionedCallPartitionedCall+bpe_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_44968383�
"bpe_layer2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0bpe_layer2_44968415bpe_layer2_44968417*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_bpe_layer2_layer_call_and_return_conditional_losses_44968414�
activation_3/PartitionedCallPartitionedCall+bpe_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_44968424�
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@l
NoOpNoOp#^bpe_layer1/StatefulPartitionedCall#^bpe_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2H
"bpe_layer1/StatefulPartitionedCall"bpe_layer1/StatefulPartitionedCall2H
"bpe_layer2/StatefulPartitionedCall"bpe_layer2/StatefulPartitionedCall:($
"
_user_specified_name
44968417:($
"
_user_specified_name
44968415:($
"
_user_specified_name
44968376:($
"
_user_specified_name
44968374:f b
4
_output_shapes"
 :������������������
*
_user_specified_namebpe_layer1_input
�
R
&__inference_add_layer_call_fn_44969634
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_44968835d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������@:���������@:UQ
+
_output_shapes
:���������@
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������@
"
_user_specified_name
inputs_0
�
�
1__inference_spatial_layer2_layer_call_fn_44969763

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_spatial_layer2_layer_call_and_return_conditional_losses_44968258s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969759:($
"
_user_specified_name
44969757:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�Y
�
C__inference_model_layer_call_and_return_conditional_losses_44968900
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layer&
bve_layer_1_44968696:@"
bve_layer_1_44968698:@1
spatial_transformation_44968723:@-
spatial_transformation_44968725:@1
spatial_transformation_44968727:@@-
spatial_transformation_44968729:@
bpe_44968732:@
bpe_44968734:@
bpe_44968736:@@
bpe_44968738:@&
bve_layer_2_44968772:@@"
bve_layer_2_44968774:@3
multi_head_attention_44968813:@@/
multi_head_attention_44968815:@3
multi_head_attention_44968817:@@/
multi_head_attention_44968819:@3
multi_head_attention_44968821:@@/
multi_head_attention_44968823:@3
multi_head_attention_44968825:@@+
multi_head_attention_44968827:@ 
dense_44968868:@@
dense_44968870:@

u_44968880:@@

u_44968882:@

u_44968884:@@

u_44968886:@#
output_u_44968896:@
identity��BPE/StatefulPartitionedCall�U/StatefulPartitionedCall�#bve_layer_1/StatefulPartitionedCall�#bve_layer_2/StatefulPartitionedCall�dense/StatefulPartitionedCall�,multi_head_attention/StatefulPartitionedCall� output_u/StatefulPartitionedCall�.spatial_transformation/StatefulPartitionedCall�
rescaling/PartitionedCallPartitionedCallx_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_rescaling_layer_call_and_return_conditional_losses_44968627�
rescaling_1/PartitionedCallPartitionedCallt_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_1_layer_call_and_return_conditional_losses_44968636�
rescaling_2/PartitionedCallPartitionedCall	xbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_2_layer_call_and_return_conditional_losses_44968646�
rescaling_3/PartitionedCallPartitionedCall	tbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_3_layer_call_and_return_conditional_losses_44968655�
concatenate/PartitionedCallPartitionedCall"rescaling/PartitionedCall:output:0$rescaling_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_44968663�
#bve_layer_1/StatefulPartitionedCallStatefulPartitionedCall	ubc_layerbve_layer_1_44968696bve_layer_1_44968698*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_bve_layer_1_layer_call_and_return_conditional_losses_44968695�
concatenate_1/PartitionedCallPartitionedCall$rescaling_2/PartitionedCall:output:0$rescaling_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *T
fORM
K__inference_concatenate_1_layer_call_and_return_conditional_losses_44968707�
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_44968721�
.spatial_transformation/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0spatial_transformation_44968723spatial_transformation_44968725spatial_transformation_44968727spatial_transformation_44968729*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968271�
BPE/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0bpe_44968732bpe_44968734bpe_44968736bpe_44968738*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_BPE_layer_call_and_return_conditional_losses_44968427�
#bve_layer_2/StatefulPartitionedCallStatefulPartitionedCall,bve_layer_1/StatefulPartitionedCall:output:0bve_layer_2_44968772bve_layer_2_44968774*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_bve_layer_2_layer_call_and_return_conditional_losses_44968771�
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:0,bve_layer_2/StatefulPartitionedCall:output:0$BPE/StatefulPartitionedCall:output:0multi_head_attention_44968813multi_head_attention_44968815multi_head_attention_44968817multi_head_attention_44968819multi_head_attention_44968821multi_head_attention_44968823multi_head_attention_44968825multi_head_attention_44968827*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44968812�
add/PartitionedCallPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:05multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_44968835�
dense/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0dense_44968868dense_44968870*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_44968867�
flatten/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_44968878�
U/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0
u_44968880
u_44968882
u_44968884
u_44968886*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_U_layer_call_and_return_conditional_losses_44968543�
 output_u/StatefulPartitionedCallStatefulPartitionedCall"U/StatefulPartitionedCall:output:0output_u_44968896*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_output_u_layer_call_and_return_conditional_losses_44968895x
IdentityIdentity)output_u/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp^BPE/StatefulPartitionedCall^U/StatefulPartitionedCall$^bve_layer_1/StatefulPartitionedCall$^bve_layer_2/StatefulPartitionedCall^dense/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall!^output_u/StatefulPartitionedCall/^spatial_transformation/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:������������������:������������������:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
BPE/StatefulPartitionedCallBPE/StatefulPartitionedCall26
U/StatefulPartitionedCallU/StatefulPartitionedCall2J
#bve_layer_1/StatefulPartitionedCall#bve_layer_1/StatefulPartitionedCall2J
#bve_layer_2/StatefulPartitionedCall#bve_layer_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2D
 output_u/StatefulPartitionedCall output_u/StatefulPartitionedCall2`
.spatial_transformation/StatefulPartitionedCall.spatial_transformation/StatefulPartitionedCall:($
"
_user_specified_name
44968896:($
"
_user_specified_name
44968886:($
"
_user_specified_name
44968884:($
"
_user_specified_name
44968882:($
"
_user_specified_name
44968880:($
"
_user_specified_name
44968870:($
"
_user_specified_name
44968868:($
"
_user_specified_name
44968827:($
"
_user_specified_name
44968825:($
"
_user_specified_name
44968823:($
"
_user_specified_name
44968821:($
"
_user_specified_name
44968819:($
"
_user_specified_name
44968817:($
"
_user_specified_name
44968815:($
"
_user_specified_name
44968813:($
"
_user_specified_name
44968774:($
"
_user_specified_name
44968772:($
"
_user_specified_name
44968738:($
"
_user_specified_name
44968736:($
"
_user_specified_name
44968734:($
"
_user_specified_name
44968732:(
$
"
_user_specified_name
44968729:(	$
"
_user_specified_name
44968727:($
"
_user_specified_name
44968725:($
"
_user_specified_name
44968723:($
"
_user_specified_name
44968698:($
"
_user_specified_name
44968696:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:���������
!
_user_specified_name	t_input:P L
'
_output_shapes
:���������
!
_user_specified_name	x_input
�
�
$__inference__traced_restore_44970237
file_prefix5
#assignvariableop_bve_layer_1_kernel:@1
#assignvariableop_1_bve_layer_1_bias:@7
%assignvariableop_2_bve_layer_2_kernel:@@1
#assignvariableop_3_bve_layer_2_bias:@1
assignvariableop_4_dense_kernel:@@+
assignvariableop_5_dense_bias:@4
"assignvariableop_6_output_u_kernel:@:
(assignvariableop_7_spatial_layer1_kernel:@4
&assignvariableop_8_spatial_layer1_bias:@:
(assignvariableop_9_spatial_layer2_kernel:@@5
'assignvariableop_10_spatial_layer2_bias:@7
%assignvariableop_11_bpe_layer1_kernel:@1
#assignvariableop_12_bpe_layer1_bias:@7
%assignvariableop_13_bpe_layer2_kernel:@@1
#assignvariableop_14_bpe_layer2_bias:@K
5assignvariableop_15_multi_head_attention_query_kernel:@@E
3assignvariableop_16_multi_head_attention_query_bias:@I
3assignvariableop_17_multi_head_attention_key_kernel:@@C
1assignvariableop_18_multi_head_attention_key_bias:@K
5assignvariableop_19_multi_head_attention_value_kernel:@@E
3assignvariableop_20_multi_head_attention_value_bias:@V
@assignvariableop_21_multi_head_attention_attention_output_kernel:@@L
>assignvariableop_22_multi_head_attention_attention_output_bias:@0
assignvariableop_23_ou1_kernel:@@*
assignvariableop_24_ou1_bias:@0
assignvariableop_25_ou2_kernel:@@*
assignvariableop_26_ou2_bias:@
identity_28��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_10�AssignVariableOp_11�AssignVariableOp_12�AssignVariableOp_13�AssignVariableOp_14�AssignVariableOp_15�AssignVariableOp_16�AssignVariableOp_17�AssignVariableOp_18�AssignVariableOp_19�AssignVariableOp_2�AssignVariableOp_20�AssignVariableOp_21�AssignVariableOp_22�AssignVariableOp_23�AssignVariableOp_24�AssignVariableOp_25�AssignVariableOp_26�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�AssignVariableOp_8�AssignVariableOp_9�

RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*�	
value�	B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*K
valueBB@B B B B B B B B B B B B B B B B B B B B B B B B B B B B �
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*�
_output_shapesr
p::::::::::::::::::::::::::::**
dtypes 
2[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOpAssignVariableOp#assignvariableop_bve_layer_1_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_1AssignVariableOp#assignvariableop_1_bve_layer_1_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_2AssignVariableOp%assignvariableop_2_bve_layer_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_3AssignVariableOp#assignvariableop_3_bve_layer_2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_4AssignVariableOpassignvariableop_4_dense_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_6AssignVariableOp"assignvariableop_6_output_u_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_7AssignVariableOp(assignvariableop_7_spatial_layer1_kernelIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_8AssignVariableOp&assignvariableop_8_spatial_layer1_biasIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_9AssignVariableOp(assignvariableop_9_spatial_layer2_kernelIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_10AssignVariableOp'assignvariableop_10_spatial_layer2_biasIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_11AssignVariableOp%assignvariableop_11_bpe_layer1_kernelIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_12AssignVariableOp#assignvariableop_12_bpe_layer1_biasIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_13AssignVariableOp%assignvariableop_13_bpe_layer2_kernelIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_14AssignVariableOp#assignvariableop_14_bpe_layer2_biasIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_15AssignVariableOp5assignvariableop_15_multi_head_attention_query_kernelIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_16AssignVariableOp3assignvariableop_16_multi_head_attention_query_biasIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_17AssignVariableOp3assignvariableop_17_multi_head_attention_key_kernelIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_18AssignVariableOp1assignvariableop_18_multi_head_attention_key_biasIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_19AssignVariableOp5assignvariableop_19_multi_head_attention_value_kernelIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_20AssignVariableOp3assignvariableop_20_multi_head_attention_value_biasIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_21AssignVariableOp@assignvariableop_21_multi_head_attention_attention_output_kernelIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_22AssignVariableOp>assignvariableop_22_multi_head_attention_attention_output_biasIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_23AssignVariableOpassignvariableop_23_ou1_kernelIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_24AssignVariableOpassignvariableop_24_ou1_biasIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_25AssignVariableOpassignvariableop_25_ou2_kernelIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:�
AssignVariableOp_26AssignVariableOpassignvariableop_26_ou2_biasIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 �
Identity_27Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_28IdentityIdentity_27:output:0^NoOp_1*
T0*
_output_shapes
: �
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_28Identity_28:output:0*(
_construction_contextkEagerRuntime*K
_input_shapes:
8: : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:($
"
_user_specified_name
ou2/bias:*&
$
_user_specified_name
ou2/kernel:($
"
_user_specified_name
ou1/bias:*&
$
_user_specified_name
ou1/kernel:JF
D
_user_specified_name,*multi_head_attention/attention_output/bias:LH
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?;
9
_user_specified_name!multi_head_attention/value/bias:A=
;
_user_specified_name#!multi_head_attention/value/kernel:=9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:/+
)
_user_specified_namebpe_layer2/bias:1-
+
_user_specified_namebpe_layer2/kernel:/+
)
_user_specified_namebpe_layer1/bias:1-
+
_user_specified_namebpe_layer1/kernel:3/
-
_user_specified_namespatial_layer2/bias:5
1
/
_user_specified_namespatial_layer2/kernel:3	/
-
_user_specified_namespatial_layer1/bias:51
/
_user_specified_namespatial_layer1/kernel:/+
)
_user_specified_nameoutput_u/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:0,
*
_user_specified_namebve_layer_2/bias:2.
,
_user_specified_namebve_layer_2/kernel:0,
*
_user_specified_namebve_layer_1/bias:2.
,
_user_specified_namebve_layer_1/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�	
�
A__inference_ou1_layer_call_and_return_conditional_losses_44968509

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
K
/__inference_activation_4_layer_call_fn_44969925

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_44968519`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
K
/__inference_activation_3_layer_call_fn_44969896

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_44968424m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������@:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
.__inference_bve_layer_2_layer_call_fn_44969479

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_bve_layer_2_layer_call_and_return_conditional_losses_44968771|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969475:($
"
_user_specified_name
44969473:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
��
�
#__inference__wrapped_model_44968185
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layerE
3model_bve_layer_1_tensordot_readvariableop_resource:@?
1model_bve_layer_1_biasadd_readvariableop_resource:@_
Mmodel_spatial_transformation_spatial_layer1_tensordot_readvariableop_resource:@Y
Kmodel_spatial_transformation_spatial_layer1_biasadd_readvariableop_resource:@_
Mmodel_spatial_transformation_spatial_layer2_tensordot_readvariableop_resource:@@Y
Kmodel_spatial_transformation_spatial_layer2_biasadd_readvariableop_resource:@H
6model_bpe_bpe_layer1_tensordot_readvariableop_resource:@B
4model_bpe_bpe_layer1_biasadd_readvariableop_resource:@H
6model_bpe_bpe_layer2_tensordot_readvariableop_resource:@@B
4model_bpe_bpe_layer2_biasadd_readvariableop_resource:@E
3model_bve_layer_2_tensordot_readvariableop_resource:@@?
1model_bve_layer_2_biasadd_readvariableop_resource:@\
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:@@N
<model_multi_head_attention_query_add_readvariableop_resource:@Z
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:@@L
:model_multi_head_attention_key_add_readvariableop_resource:@\
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:@@N
<model_multi_head_attention_value_add_readvariableop_resource:@g
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@U
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:@?
-model_dense_tensordot_readvariableop_resource:@@9
+model_dense_biasadd_readvariableop_resource:@<
*model_u_ou1_matmul_readvariableop_resource:@@9
+model_u_ou1_biasadd_readvariableop_resource:@<
*model_u_ou2_matmul_readvariableop_resource:@@9
+model_u_ou2_biasadd_readvariableop_resource:@?
-model_output_u_matmul_readvariableop_resource:@
identity��+model/BPE/bpe_layer1/BiasAdd/ReadVariableOp�-model/BPE/bpe_layer1/Tensordot/ReadVariableOp�+model/BPE/bpe_layer2/BiasAdd/ReadVariableOp�-model/BPE/bpe_layer2/Tensordot/ReadVariableOp�"model/U/ou1/BiasAdd/ReadVariableOp�!model/U/ou1/MatMul/ReadVariableOp�"model/U/ou2/BiasAdd/ReadVariableOp�!model/U/ou2/MatMul/ReadVariableOp�(model/bve_layer_1/BiasAdd/ReadVariableOp�*model/bve_layer_1/Tensordot/ReadVariableOp�(model/bve_layer_2/BiasAdd/ReadVariableOp�*model/bve_layer_2/Tensordot/ReadVariableOp�"model/dense/BiasAdd/ReadVariableOp�$model/dense/Tensordot/ReadVariableOp�>model/multi_head_attention/attention_output/add/ReadVariableOp�Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp�1model/multi_head_attention/key/add/ReadVariableOp�;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp�3model/multi_head_attention/query/add/ReadVariableOp�=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp�3model/multi_head_attention/value/add/ReadVariableOp�=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp�$model/output_u/MatMul/ReadVariableOp�Bmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOp�Dmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOp�Bmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOp�Dmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp[
model/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @]
model/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ��v
model/rescaling/mulMulx_inputmodel/rescaling/Cast/x:output:0*
T0*'
_output_shapes
:����������
model/rescaling/addAddV2model/rescaling/mul:z:0!model/rescaling/Cast_1/x:output:0*
T0*'
_output_shapes
:���������]
model/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
model/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ��z
model/rescaling_1/mulMult_input!model/rescaling_1/Cast/x:output:0*
T0*'
_output_shapes
:����������
model/rescaling_1/addAddV2model/rescaling_1/mul:z:0#model/rescaling_1/Cast_1/x:output:0*
T0*'
_output_shapes
:���������]
model/rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @e
model/rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������u
model/rescaling_2/Cast_1Cast#model/rescaling_2/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: �
model/rescaling_2/mulMul	xbc_layer!model/rescaling_2/Cast/x:output:0*
T0*4
_output_shapes"
 :�������������������
model/rescaling_2/addAddV2model/rescaling_2/mul:z:0model/rescaling_2/Cast_1:y:0*
T0*4
_output_shapes"
 :������������������]
model/rescaling_3/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
model/rescaling_3/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ���
model/rescaling_3/mulMul	tbc_layer!model/rescaling_3/Cast/x:output:0*
T0*4
_output_shapes"
 :�������������������
model/rescaling_3/addAddV2model/rescaling_3/mul:z:0#model/rescaling_3/Cast_1/x:output:0*
T0*4
_output_shapes"
 :������������������_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate/concatConcatV2model/rescaling/add:z:0model/rescaling_1/add:z:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:����������
*model/bve_layer_1/Tensordot/ReadVariableOpReadVariableOp3model_bve_layer_1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0j
 model/bve_layer_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
 model/bve_layer_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       h
!model/bve_layer_1/Tensordot/ShapeShape	ubc_layer*
T0*
_output_shapes
::��k
)model/bve_layer_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$model/bve_layer_1/Tensordot/GatherV2GatherV2*model/bve_layer_1/Tensordot/Shape:output:0)model/bve_layer_1/Tensordot/free:output:02model/bve_layer_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+model/bve_layer_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&model/bve_layer_1/Tensordot/GatherV2_1GatherV2*model/bve_layer_1/Tensordot/Shape:output:0)model/bve_layer_1/Tensordot/axes:output:04model/bve_layer_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!model/bve_layer_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
 model/bve_layer_1/Tensordot/ProdProd-model/bve_layer_1/Tensordot/GatherV2:output:0*model/bve_layer_1/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#model/bve_layer_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
"model/bve_layer_1/Tensordot/Prod_1Prod/model/bve_layer_1/Tensordot/GatherV2_1:output:0,model/bve_layer_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'model/bve_layer_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model/bve_layer_1/Tensordot/concatConcatV2)model/bve_layer_1/Tensordot/free:output:0)model/bve_layer_1/Tensordot/axes:output:00model/bve_layer_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
!model/bve_layer_1/Tensordot/stackPack)model/bve_layer_1/Tensordot/Prod:output:0+model/bve_layer_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
%model/bve_layer_1/Tensordot/transpose	Transpose	ubc_layer+model/bve_layer_1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
#model/bve_layer_1/Tensordot/ReshapeReshape)model/bve_layer_1/Tensordot/transpose:y:0*model/bve_layer_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
"model/bve_layer_1/Tensordot/MatMulMatMul,model/bve_layer_1/Tensordot/Reshape:output:02model/bve_layer_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@m
#model/bve_layer_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@k
)model/bve_layer_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$model/bve_layer_1/Tensordot/concat_1ConcatV2-model/bve_layer_1/Tensordot/GatherV2:output:0,model/bve_layer_1/Tensordot/Const_2:output:02model/bve_layer_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/bve_layer_1/TensordotReshape,model/bve_layer_1/Tensordot/MatMul:product:0-model/bve_layer_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@�
(model/bve_layer_1/BiasAdd/ReadVariableOpReadVariableOp1model_bve_layer_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/bve_layer_1/BiasAddBiasAdd$model/bve_layer_1/Tensordot:output:00model/bve_layer_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@�
model/bve_layer_1/TanhTanh"model/bve_layer_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@a
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
model/concatenate_1/concatConcatV2model/rescaling_2/add:z:0model/rescaling_3/add:z:0(model/concatenate_1/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������r
model/reshape/ShapeShape!model/concatenate/concat:output:0*
T0*
_output_shapes
::��k
!model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: m
#model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:m
#model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
model/reshape/strided_sliceStridedSlicemodel/reshape/Shape:output:0*model/reshape/strided_slice/stack:output:0,model/reshape/strided_slice/stack_1:output:0,model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask_
model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :h
model/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:�
model/reshape/ReshapeReshape!model/concatenate/concat:output:0$model/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:����������
Dmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpReadVariableOpMmodel_spatial_transformation_spatial_layer1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0�
:model/spatial_transformation/spatial_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
:model/spatial_transformation/spatial_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
;model/spatial_transformation/spatial_layer1/Tensordot/ShapeShapemodel/reshape/Reshape:output:0*
T0*
_output_shapes
::���
Cmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>model/spatial_transformation/spatial_layer1/Tensordot/GatherV2GatherV2Dmodel/spatial_transformation/spatial_layer1/Tensordot/Shape:output:0Cmodel/spatial_transformation/spatial_layer1/Tensordot/free:output:0Lmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Emodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
@model/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1GatherV2Dmodel/spatial_transformation/spatial_layer1/Tensordot/Shape:output:0Cmodel/spatial_transformation/spatial_layer1/Tensordot/axes:output:0Nmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
;model/spatial_transformation/spatial_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
:model/spatial_transformation/spatial_layer1/Tensordot/ProdProdGmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2:output:0Dmodel/spatial_transformation/spatial_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: �
=model/spatial_transformation/spatial_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
<model/spatial_transformation/spatial_layer1/Tensordot/Prod_1ProdImodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1:output:0Fmodel/spatial_transformation/spatial_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Amodel/spatial_transformation/spatial_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<model/spatial_transformation/spatial_layer1/Tensordot/concatConcatV2Cmodel/spatial_transformation/spatial_layer1/Tensordot/free:output:0Cmodel/spatial_transformation/spatial_layer1/Tensordot/axes:output:0Jmodel/spatial_transformation/spatial_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
;model/spatial_transformation/spatial_layer1/Tensordot/stackPackCmodel/spatial_transformation/spatial_layer1/Tensordot/Prod:output:0Emodel/spatial_transformation/spatial_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
?model/spatial_transformation/spatial_layer1/Tensordot/transpose	Transposemodel/reshape/Reshape:output:0Emodel/spatial_transformation/spatial_layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:����������
=model/spatial_transformation/spatial_layer1/Tensordot/ReshapeReshapeCmodel/spatial_transformation/spatial_layer1/Tensordot/transpose:y:0Dmodel/spatial_transformation/spatial_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<model/spatial_transformation/spatial_layer1/Tensordot/MatMulMatMulFmodel/spatial_transformation/spatial_layer1/Tensordot/Reshape:output:0Lmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
=model/spatial_transformation/spatial_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@�
Cmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>model/spatial_transformation/spatial_layer1/Tensordot/concat_1ConcatV2Gmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2:output:0Fmodel/spatial_transformation/spatial_layer1/Tensordot/Const_2:output:0Lmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
5model/spatial_transformation/spatial_layer1/TensordotReshapeFmodel/spatial_transformation/spatial_layer1/Tensordot/MatMul:product:0Gmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@�
Bmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpReadVariableOpKmodel_spatial_transformation_spatial_layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
3model/spatial_transformation/spatial_layer1/BiasAddBiasAdd>model/spatial_transformation/spatial_layer1/Tensordot:output:0Jmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@�
,model/spatial_transformation/activation/TanhTanh<model/spatial_transformation/spatial_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:���������@�
Dmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOpReadVariableOpMmodel_spatial_transformation_spatial_layer2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0�
:model/spatial_transformation/spatial_layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:�
:model/spatial_transformation/spatial_layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
;model/spatial_transformation/spatial_layer2/Tensordot/ShapeShape0model/spatial_transformation/activation/Tanh:y:0*
T0*
_output_shapes
::���
Cmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>model/spatial_transformation/spatial_layer2/Tensordot/GatherV2GatherV2Dmodel/spatial_transformation/spatial_layer2/Tensordot/Shape:output:0Cmodel/spatial_transformation/spatial_layer2/Tensordot/free:output:0Lmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
Emodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
@model/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1GatherV2Dmodel/spatial_transformation/spatial_layer2/Tensordot/Shape:output:0Cmodel/spatial_transformation/spatial_layer2/Tensordot/axes:output:0Nmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:�
;model/spatial_transformation/spatial_layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
:model/spatial_transformation/spatial_layer2/Tensordot/ProdProdGmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2:output:0Dmodel/spatial_transformation/spatial_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: �
=model/spatial_transformation/spatial_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
<model/spatial_transformation/spatial_layer2/Tensordot/Prod_1ProdImodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1:output:0Fmodel/spatial_transformation/spatial_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: �
Amodel/spatial_transformation/spatial_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
<model/spatial_transformation/spatial_layer2/Tensordot/concatConcatV2Cmodel/spatial_transformation/spatial_layer2/Tensordot/free:output:0Cmodel/spatial_transformation/spatial_layer2/Tensordot/axes:output:0Jmodel/spatial_transformation/spatial_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
;model/spatial_transformation/spatial_layer2/Tensordot/stackPackCmodel/spatial_transformation/spatial_layer2/Tensordot/Prod:output:0Emodel/spatial_transformation/spatial_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
?model/spatial_transformation/spatial_layer2/Tensordot/transpose	Transpose0model/spatial_transformation/activation/Tanh:y:0Emodel/spatial_transformation/spatial_layer2/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
=model/spatial_transformation/spatial_layer2/Tensordot/ReshapeReshapeCmodel/spatial_transformation/spatial_layer2/Tensordot/transpose:y:0Dmodel/spatial_transformation/spatial_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
<model/spatial_transformation/spatial_layer2/Tensordot/MatMulMatMulFmodel/spatial_transformation/spatial_layer2/Tensordot/Reshape:output:0Lmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
=model/spatial_transformation/spatial_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@�
Cmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
>model/spatial_transformation/spatial_layer2/Tensordot/concat_1ConcatV2Gmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2:output:0Fmodel/spatial_transformation/spatial_layer2/Tensordot/Const_2:output:0Lmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
5model/spatial_transformation/spatial_layer2/TensordotReshapeFmodel/spatial_transformation/spatial_layer2/Tensordot/MatMul:product:0Gmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@�
Bmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpReadVariableOpKmodel_spatial_transformation_spatial_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
3model/spatial_transformation/spatial_layer2/BiasAddBiasAdd>model/spatial_transformation/spatial_layer2/Tensordot:output:0Jmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@�
.model/spatial_transformation/activation_1/TanhTanh<model/spatial_transformation/spatial_layer2/BiasAdd:output:0*
T0*+
_output_shapes
:���������@�
-model/BPE/bpe_layer1/Tensordot/ReadVariableOpReadVariableOp6model_bpe_bpe_layer1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0m
#model/BPE/bpe_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#model/BPE/bpe_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
$model/BPE/bpe_layer1/Tensordot/ShapeShape#model/concatenate_1/concat:output:0*
T0*
_output_shapes
::��n
,model/BPE/bpe_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'model/BPE/bpe_layer1/Tensordot/GatherV2GatherV2-model/BPE/bpe_layer1/Tensordot/Shape:output:0,model/BPE/bpe_layer1/Tensordot/free:output:05model/BPE/bpe_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.model/BPE/bpe_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)model/BPE/bpe_layer1/Tensordot/GatherV2_1GatherV2-model/BPE/bpe_layer1/Tensordot/Shape:output:0,model/BPE/bpe_layer1/Tensordot/axes:output:07model/BPE/bpe_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$model/BPE/bpe_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#model/BPE/bpe_layer1/Tensordot/ProdProd0model/BPE/bpe_layer1/Tensordot/GatherV2:output:0-model/BPE/bpe_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&model/BPE/bpe_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%model/BPE/bpe_layer1/Tensordot/Prod_1Prod2model/BPE/bpe_layer1/Tensordot/GatherV2_1:output:0/model/BPE/bpe_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*model/BPE/bpe_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%model/BPE/bpe_layer1/Tensordot/concatConcatV2,model/BPE/bpe_layer1/Tensordot/free:output:0,model/BPE/bpe_layer1/Tensordot/axes:output:03model/BPE/bpe_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$model/BPE/bpe_layer1/Tensordot/stackPack,model/BPE/bpe_layer1/Tensordot/Prod:output:0.model/BPE/bpe_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(model/BPE/bpe_layer1/Tensordot/transpose	Transpose#model/concatenate_1/concat:output:0.model/BPE/bpe_layer1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
&model/BPE/bpe_layer1/Tensordot/ReshapeReshape,model/BPE/bpe_layer1/Tensordot/transpose:y:0-model/BPE/bpe_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%model/BPE/bpe_layer1/Tensordot/MatMulMatMul/model/BPE/bpe_layer1/Tensordot/Reshape:output:05model/BPE/bpe_layer1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
&model/BPE/bpe_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,model/BPE/bpe_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'model/BPE/bpe_layer1/Tensordot/concat_1ConcatV20model/BPE/bpe_layer1/Tensordot/GatherV2:output:0/model/BPE/bpe_layer1/Tensordot/Const_2:output:05model/BPE/bpe_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/BPE/bpe_layer1/TensordotReshape/model/BPE/bpe_layer1/Tensordot/MatMul:product:00model/BPE/bpe_layer1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@�
+model/BPE/bpe_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_bpe_bpe_layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/BPE/bpe_layer1/BiasAddBiasAdd'model/BPE/bpe_layer1/Tensordot:output:03model/BPE/bpe_layer1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@�
model/BPE/activation_2/TanhTanh%model/BPE/bpe_layer1/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@�
-model/BPE/bpe_layer2/Tensordot/ReadVariableOpReadVariableOp6model_bpe_bpe_layer2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0m
#model/BPE/bpe_layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#model/BPE/bpe_layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       �
$model/BPE/bpe_layer2/Tensordot/ShapeShapemodel/BPE/activation_2/Tanh:y:0*
T0*
_output_shapes
::��n
,model/BPE/bpe_layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'model/BPE/bpe_layer2/Tensordot/GatherV2GatherV2-model/BPE/bpe_layer2/Tensordot/Shape:output:0,model/BPE/bpe_layer2/Tensordot/free:output:05model/BPE/bpe_layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.model/BPE/bpe_layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
)model/BPE/bpe_layer2/Tensordot/GatherV2_1GatherV2-model/BPE/bpe_layer2/Tensordot/Shape:output:0,model/BPE/bpe_layer2/Tensordot/axes:output:07model/BPE/bpe_layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$model/BPE/bpe_layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
#model/BPE/bpe_layer2/Tensordot/ProdProd0model/BPE/bpe_layer2/Tensordot/GatherV2:output:0-model/BPE/bpe_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&model/BPE/bpe_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
%model/BPE/bpe_layer2/Tensordot/Prod_1Prod2model/BPE/bpe_layer2/Tensordot/GatherV2_1:output:0/model/BPE/bpe_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*model/BPE/bpe_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
%model/BPE/bpe_layer2/Tensordot/concatConcatV2,model/BPE/bpe_layer2/Tensordot/free:output:0,model/BPE/bpe_layer2/Tensordot/axes:output:03model/BPE/bpe_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
$model/BPE/bpe_layer2/Tensordot/stackPack,model/BPE/bpe_layer2/Tensordot/Prod:output:0.model/BPE/bpe_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
(model/BPE/bpe_layer2/Tensordot/transpose	Transposemodel/BPE/activation_2/Tanh:y:0.model/BPE/bpe_layer2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :������������������@�
&model/BPE/bpe_layer2/Tensordot/ReshapeReshape,model/BPE/bpe_layer2/Tensordot/transpose:y:0-model/BPE/bpe_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
%model/BPE/bpe_layer2/Tensordot/MatMulMatMul/model/BPE/bpe_layer2/Tensordot/Reshape:output:05model/BPE/bpe_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@p
&model/BPE/bpe_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,model/BPE/bpe_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
'model/BPE/bpe_layer2/Tensordot/concat_1ConcatV20model/BPE/bpe_layer2/Tensordot/GatherV2:output:0/model/BPE/bpe_layer2/Tensordot/Const_2:output:05model/BPE/bpe_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/BPE/bpe_layer2/TensordotReshape/model/BPE/bpe_layer2/Tensordot/MatMul:product:00model/BPE/bpe_layer2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@�
+model/BPE/bpe_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_bpe_bpe_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/BPE/bpe_layer2/BiasAddBiasAdd'model/BPE/bpe_layer2/Tensordot:output:03model/BPE/bpe_layer2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@�
model/BPE/activation_3/TanhTanh%model/BPE/bpe_layer2/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@�
*model/bve_layer_2/Tensordot/ReadVariableOpReadVariableOp3model_bve_layer_2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0j
 model/bve_layer_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:q
 model/bve_layer_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
!model/bve_layer_2/Tensordot/ShapeShapemodel/bve_layer_1/Tanh:y:0*
T0*
_output_shapes
::��k
)model/bve_layer_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$model/bve_layer_2/Tensordot/GatherV2GatherV2*model/bve_layer_2/Tensordot/Shape:output:0)model/bve_layer_2/Tensordot/free:output:02model/bve_layer_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:m
+model/bve_layer_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
&model/bve_layer_2/Tensordot/GatherV2_1GatherV2*model/bve_layer_2/Tensordot/Shape:output:0)model/bve_layer_2/Tensordot/axes:output:04model/bve_layer_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:k
!model/bve_layer_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
 model/bve_layer_2/Tensordot/ProdProd-model/bve_layer_2/Tensordot/GatherV2:output:0*model/bve_layer_2/Tensordot/Const:output:0*
T0*
_output_shapes
: m
#model/bve_layer_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
"model/bve_layer_2/Tensordot/Prod_1Prod/model/bve_layer_2/Tensordot/GatherV2_1:output:0,model/bve_layer_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: i
'model/bve_layer_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
"model/bve_layer_2/Tensordot/concatConcatV2)model/bve_layer_2/Tensordot/free:output:0)model/bve_layer_2/Tensordot/axes:output:00model/bve_layer_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
!model/bve_layer_2/Tensordot/stackPack)model/bve_layer_2/Tensordot/Prod:output:0+model/bve_layer_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
%model/bve_layer_2/Tensordot/transpose	Transposemodel/bve_layer_1/Tanh:y:0+model/bve_layer_2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :������������������@�
#model/bve_layer_2/Tensordot/ReshapeReshape)model/bve_layer_2/Tensordot/transpose:y:0*model/bve_layer_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
"model/bve_layer_2/Tensordot/MatMulMatMul,model/bve_layer_2/Tensordot/Reshape:output:02model/bve_layer_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@m
#model/bve_layer_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@k
)model/bve_layer_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
$model/bve_layer_2/Tensordot/concat_1ConcatV2-model/bve_layer_2/Tensordot/GatherV2:output:0,model/bve_layer_2/Tensordot/Const_2:output:02model/bve_layer_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/bve_layer_2/TensordotReshape,model/bve_layer_2/Tensordot/MatMul:product:0-model/bve_layer_2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@�
(model/bve_layer_2/BiasAdd/ReadVariableOpReadVariableOp1model_bve_layer_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/bve_layer_2/BiasAddBiasAdd$model/bve_layer_2/Tensordot:output:00model/bve_layer_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@�
model/bve_layer_2/TanhTanh"model/bve_layer_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@�
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
.model/multi_head_attention/query/einsum/EinsumEinsum2model/spatial_transformation/activation_1/Tanh:y:0Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abde�
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
(model/multi_head_attention/query/add/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
,model/multi_head_attention/key/einsum/EinsumEinsummodel/BPE/activation_3/Tanh:y:0Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
&model/multi_head_attention/key/add/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
.model/multi_head_attention/value/einsum/EinsumEinsummodel/bve_layer_2/Tanh:y:0Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abde�
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
(model/multi_head_attention/value/add/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@e
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >�
model/multi_head_attention/MulMul,model/multi_head_attention/query/add/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:���������@�
(model/multi_head_attention/einsum/EinsumEinsum*model/multi_head_attention/key/add/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*8
_output_shapes&
$:"������������������*
equationaecd,abcd->acbe�
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*8
_output_shapes&
$:"�������������������
+model/multi_head_attention/dropout/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"�������������������
*model/multi_head_attention/einsum_1/EinsumEinsum4model/multi_head_attention/dropout/Identity:output:0,model/multi_head_attention/value/add/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������@*
equationabcd,cde->abe�
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
3model/multi_head_attention/attention_output/add/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@�
model/add/addAddV22model/spatial_transformation/activation_1/Tanh:y:07model/multi_head_attention/attention_output/add/add:z:0*
T0*+
_output_shapes
:���������@�
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0d
model/dense/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:k
model/dense/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       j
model/dense/Tensordot/ShapeShapemodel/add/add:z:0*
T0*
_output_shapes
::��e
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense/Tensordot/GatherV2GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/free:output:0,model/dense/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
%model/dense/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
 model/dense/Tensordot/GatherV2_1GatherV2$model/dense/Tensordot/Shape:output:0#model/dense/Tensordot/axes:output:0.model/dense/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:e
model/dense/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: �
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: �
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
model/dense/Tensordot/transpose	Transposemodel/add/add:z:0%model/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@g
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@�
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@l
model/dense/TanhTanhmodel/dense/BiasAdd:output:0*
T0*+
_output_shapes
:���������@d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   �
model/flatten/ReshapeReshapemodel/dense/Tanh:y:0model/flatten/Const:output:0*
T0*'
_output_shapes
:���������@�
!model/U/ou1/MatMul/ReadVariableOpReadVariableOp*model_u_ou1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/U/ou1/MatMulMatMulmodel/flatten/Reshape:output:0)model/U/ou1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/U/ou1/BiasAdd/ReadVariableOpReadVariableOp+model_u_ou1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/U/ou1/BiasAddBiasAddmodel/U/ou1/MatMul:product:0*model/U/ou1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@q
model/U/activation_4/TanhTanhmodel/U/ou1/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
!model/U/ou2/MatMul/ReadVariableOpReadVariableOp*model_u_ou2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0�
model/U/ou2/MatMulMatMulmodel/U/activation_4/Tanh:y:0)model/U/ou2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@�
"model/U/ou2/BiasAdd/ReadVariableOpReadVariableOp+model_u_ou2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
model/U/ou2/BiasAddBiasAddmodel/U/ou2/MatMul:product:0*model/U/ou2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@q
model/U/activation_5/TanhTanhmodel/U/ou2/BiasAdd:output:0*
T0*'
_output_shapes
:���������@�
$model/output_u/MatMul/ReadVariableOpReadVariableOp-model_output_u_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0�
model/output_u/MatMulMatMulmodel/U/activation_5/Tanh:y:0,model/output_u/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������n
IdentityIdentitymodel/output_u/MatMul:product:0^NoOp*
T0*'
_output_shapes
:����������
NoOpNoOp,^model/BPE/bpe_layer1/BiasAdd/ReadVariableOp.^model/BPE/bpe_layer1/Tensordot/ReadVariableOp,^model/BPE/bpe_layer2/BiasAdd/ReadVariableOp.^model/BPE/bpe_layer2/Tensordot/ReadVariableOp#^model/U/ou1/BiasAdd/ReadVariableOp"^model/U/ou1/MatMul/ReadVariableOp#^model/U/ou2/BiasAdd/ReadVariableOp"^model/U/ou2/MatMul/ReadVariableOp)^model/bve_layer_1/BiasAdd/ReadVariableOp+^model/bve_layer_1/Tensordot/ReadVariableOp)^model/bve_layer_2/BiasAdd/ReadVariableOp+^model/bve_layer_2/Tensordot/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOp%^model/output_u/MatMul/ReadVariableOpC^model/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpE^model/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpC^model/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpE^model/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:������������������:������������������:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+model/BPE/bpe_layer1/BiasAdd/ReadVariableOp+model/BPE/bpe_layer1/BiasAdd/ReadVariableOp2^
-model/BPE/bpe_layer1/Tensordot/ReadVariableOp-model/BPE/bpe_layer1/Tensordot/ReadVariableOp2Z
+model/BPE/bpe_layer2/BiasAdd/ReadVariableOp+model/BPE/bpe_layer2/BiasAdd/ReadVariableOp2^
-model/BPE/bpe_layer2/Tensordot/ReadVariableOp-model/BPE/bpe_layer2/Tensordot/ReadVariableOp2H
"model/U/ou1/BiasAdd/ReadVariableOp"model/U/ou1/BiasAdd/ReadVariableOp2F
!model/U/ou1/MatMul/ReadVariableOp!model/U/ou1/MatMul/ReadVariableOp2H
"model/U/ou2/BiasAdd/ReadVariableOp"model/U/ou2/BiasAdd/ReadVariableOp2F
!model/U/ou2/MatMul/ReadVariableOp!model/U/ou2/MatMul/ReadVariableOp2T
(model/bve_layer_1/BiasAdd/ReadVariableOp(model/bve_layer_1/BiasAdd/ReadVariableOp2X
*model/bve_layer_1/Tensordot/ReadVariableOp*model/bve_layer_1/Tensordot/ReadVariableOp2T
(model/bve_layer_2/BiasAdd/ReadVariableOp(model/bve_layer_2/BiasAdd/ReadVariableOp2X
*model/bve_layer_2/Tensordot/ReadVariableOp*model/bve_layer_2/Tensordot/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2�
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2�
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp2L
$model/output_u/MatMul/ReadVariableOp$model/output_u/MatMul/ReadVariableOp2�
Bmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpBmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOp2�
Dmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpDmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOp2�
Bmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpBmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOp2�
Dmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOpDmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:���������
!
_user_specified_name	t_input:P L
'
_output_shapes
:���������
!
_user_specified_name	x_input
�
�
.__inference_bve_layer_1_layer_call_fn_44969439

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_bve_layer_1_layer_call_and_return_conditional_losses_44968695|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969435:($
"
_user_specified_name
44969433:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
�
L__inference_spatial_layer1_layer_call_and_return_conditional_losses_44969744

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
u
I__inference_concatenate_layer_call_and_return_conditional_losses_44969372
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :w
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
-__inference_bpe_layer2_layer_call_fn_44969861

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_bpe_layer2_layer_call_and_return_conditional_losses_44968414|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969857:($
"
_user_specified_name
44969855:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
f
J__inference_activation_4_layer_call_and_return_conditional_losses_44969930

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:���������@P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

a
E__inference_reshape_layer_call_and_return_conditional_losses_44968721

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Z
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
9__inference_spatial_transformation_layer_call_fn_44968300
spatial_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *]
fXRV
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968271s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44968296:($
"
_user_specified_name
44968294:($
"
_user_specified_name
44968292:($
"
_user_specified_name
44968290:a ]
+
_output_shapes
:���������
.
_user_specified_namespatial_layer1_input
�
�
(__inference_model_layer_call_fn_44969078
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layer
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@
	unknown_6:@
	unknown_7:@@
	unknown_8:@
	unknown_9:@@

unknown_10:@ 

unknown_11:@@

unknown_12:@ 

unknown_13:@@

unknown_14:@ 

unknown_15:@@

unknown_16:@ 

unknown_17:@@

unknown_18:@

unknown_19:@@

unknown_20:@

unknown_21:@@

unknown_22:@

unknown_23:@@

unknown_24:@

unknown_25:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallx_inputt_input	xbc_layer	tbc_layer	ubc_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18
unknown_19
unknown_20
unknown_21
unknown_22
unknown_23
unknown_24
unknown_25*+
Tin$
"2 *
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*=
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8� *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_44968900o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*�
_input_shapes�
�:���������:���������:������������������:������������������:������������������: : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969074:($
"
_user_specified_name
44969072:($
"
_user_specified_name
44969070:($
"
_user_specified_name
44969068:($
"
_user_specified_name
44969066:($
"
_user_specified_name
44969064:($
"
_user_specified_name
44969062:($
"
_user_specified_name
44969060:($
"
_user_specified_name
44969058:($
"
_user_specified_name
44969056:($
"
_user_specified_name
44969054:($
"
_user_specified_name
44969052:($
"
_user_specified_name
44969050:($
"
_user_specified_name
44969048:($
"
_user_specified_name
44969046:($
"
_user_specified_name
44969044:($
"
_user_specified_name
44969042:($
"
_user_specified_name
44969040:($
"
_user_specified_name
44969038:($
"
_user_specified_name
44969036:($
"
_user_specified_name
44969034:(
$
"
_user_specified_name
44969032:(	$
"
_user_specified_name
44969030:($
"
_user_specified_name
44969028:($
"
_user_specified_name
44969026:($
"
_user_specified_name
44969024:($
"
_user_specified_name
44969022:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :������������������
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:���������
!
_user_specified_name	t_input:P L
'
_output_shapes
:���������
!
_user_specified_name	x_input
�
�
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968271
spatial_layer1_input)
spatial_layer1_44968218:@%
spatial_layer1_44968220:@)
spatial_layer2_44968259:@@%
spatial_layer2_44968261:@
identity��&spatial_layer1/StatefulPartitionedCall�&spatial_layer2/StatefulPartitionedCall�
&spatial_layer1/StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputspatial_layer1_44968218spatial_layer1_44968220*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_spatial_layer1_layer_call_and_return_conditional_losses_44968217�
activation/PartitionedCallPartitionedCall/spatial_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_44968227�
&spatial_layer2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0spatial_layer2_44968259spatial_layer2_44968261*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_spatial_layer2_layer_call_and_return_conditional_losses_44968258�
activation_1/PartitionedCallPartitionedCall/spatial_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_44968268x
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@t
NoOpNoOp'^spatial_layer1/StatefulPartitionedCall'^spatial_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2P
&spatial_layer1/StatefulPartitionedCall&spatial_layer1/StatefulPartitionedCall2P
&spatial_layer2/StatefulPartitionedCall&spatial_layer2/StatefulPartitionedCall:($
"
_user_specified_name
44968261:($
"
_user_specified_name
44968259:($
"
_user_specified_name
44968220:($
"
_user_specified_name
44968218:a ]
+
_output_shapes
:���������
.
_user_specified_namespatial_layer1_input
�
K
/__inference_activation_5_layer_call_fn_44969954

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_44968540`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_activation_1_layer_call_and_return_conditional_losses_44969803

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:���������@T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
$__inference_U_layer_call_fn_44968585
	ou1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	ou1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_U_layer_call_and_return_conditional_losses_44968559o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44968581:($
"
_user_specified_name
44968579:($
"
_user_specified_name
44968577:($
"
_user_specified_name
44968575:R N
'
_output_shapes
:���������@
#
_user_specified_name	ou1_input
�
�
&__inference_ou1_layer_call_fn_44969910

inputs
unknown:@@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_ou1_layer_call_and_return_conditional_losses_44968509o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969906:($
"
_user_specified_name
44969904:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_activation_3_layer_call_and_return_conditional_losses_44969901

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :������������������@]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������@:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
d
H__inference_activation_layer_call_and_return_conditional_losses_44968227

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:���������@T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_activation_1_layer_call_and_return_conditional_losses_44968268

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:���������@T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
I__inference_bve_layer_1_layer_call_and_return_conditional_losses_44969470

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@]
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@d
IdentityIdentityTanh:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�.
�
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44968812	
query	
value
keyA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"������������������*
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"������������������z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"�������������������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������@*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������@:������������������@:������������������@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:YU
4
_output_shapes"
 :������������������@

_user_specified_namekey:[W
4
_output_shapes"
 :������������������@

_user_specified_namevalue:R N
+
_output_shapes
:���������@

_user_specified_namequery
�
s
I__inference_concatenate_layer_call_and_return_conditional_losses_44968663

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :u
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:���������W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:OK
'
_output_shapes
:���������
 
_user_specified_nameinputs:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_44968878

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
?__inference_U_layer_call_and_return_conditional_losses_44968559
	ou1_input
ou1_44968546:@@
ou1_44968548:@
ou2_44968552:@@
ou2_44968554:@
identity��ou1/StatefulPartitionedCall�ou2/StatefulPartitionedCall�
ou1/StatefulPartitionedCallStatefulPartitionedCall	ou1_inputou1_44968546ou1_44968548*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_ou1_layer_call_and_return_conditional_losses_44968509�
activation_4/PartitionedCallPartitionedCall$ou1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_4_layer_call_and_return_conditional_losses_44968519�
ou2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0ou2_44968552ou2_44968554*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_ou2_layer_call_and_return_conditional_losses_44968530�
activation_5/PartitionedCallPartitionedCall$ou2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_5_layer_call_and_return_conditional_losses_44968540t
IdentityIdentity%activation_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@^
NoOpNoOp^ou1/StatefulPartitionedCall^ou2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 2:
ou1/StatefulPartitionedCallou1/StatefulPartitionedCall2:
ou2/StatefulPartitionedCallou2/StatefulPartitionedCall:($
"
_user_specified_name
44968554:($
"
_user_specified_name
44968552:($
"
_user_specified_name
44968548:($
"
_user_specified_name
44968546:R N
'
_output_shapes
:���������@
#
_user_specified_name	ou1_input
�.
�
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44968977	
query	
value
keyA
+query_einsum_einsum_readvariableop_resource:@@3
!query_add_readvariableop_resource:@?
)key_einsum_einsum_readvariableop_resource:@@1
key_add_readvariableop_resource:@A
+value_einsum_einsum_readvariableop_resource:@@3
!value_add_readvariableop_resource:@L
6attention_output_einsum_einsum_readvariableop_resource:@@:
,attention_output_add_readvariableop_resource:@
identity��#attention_output/add/ReadVariableOp�-attention_output/einsum/Einsum/ReadVariableOp�key/add/ReadVariableOp� key/einsum/Einsum/ReadVariableOp�query/add/ReadVariableOp�"query/einsum/Einsum/ReadVariableOp�value/add/ReadVariableOp�"value/einsum/Einsum/ReadVariableOp�
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:���������@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:���������@�
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@�
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"������������������@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0�
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"������������������@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:���������@�
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"������������������*
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"������������������z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"�������������������
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:���������@*
equationacbe,aecd->abcd�
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0�
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:���������@*
equationabcd,cde->abe�
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0�
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:���������@�
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������@:������������������@:������������������@: : : : : : : : 2J
#attention_output/add/ReadVariableOp#attention_output/add/ReadVariableOp2^
-attention_output/einsum/Einsum/ReadVariableOp-attention_output/einsum/Einsum/ReadVariableOp20
key/add/ReadVariableOpkey/add/ReadVariableOp2D
 key/einsum/Einsum/ReadVariableOp key/einsum/Einsum/ReadVariableOp24
query/add/ReadVariableOpquery/add/ReadVariableOp2H
"query/einsum/Einsum/ReadVariableOp"query/einsum/Einsum/ReadVariableOp24
value/add/ReadVariableOpvalue/add/ReadVariableOp2H
"value/einsum/Einsum/ReadVariableOp"value/einsum/Einsum/ReadVariableOp:(
$
"
_user_specified_name
resource:(	$
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:YU
4
_output_shapes"
 :������������������@

_user_specified_namekey:[W
4
_output_shapes"
 :������������������@

_user_specified_namevalue:R N
+
_output_shapes
:���������@

_user_specified_namequery
�
�
F__inference_output_u_layer_call_and_return_conditional_losses_44968895

inputs0
matmul_readvariableop_resource:@
identity��MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������:
NoOpNoOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������@: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
1__inference_spatial_layer1_layer_call_fn_44969714

inputs
unknown:@
	unknown_0:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_spatial_layer1_layer_call_and_return_conditional_losses_44968217s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969710:($
"
_user_specified_name
44969708:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�	
�
&__inference_BPE_layer_call_fn_44968456
bpe_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbpe_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_BPE_layer_call_and_return_conditional_losses_44968427|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44968452:($
"
_user_specified_name
44968450:($
"
_user_specified_name
44968448:($
"
_user_specified_name
44968446:f b
4
_output_shapes"
 :������������������
*
_user_specified_namebpe_layer1_input
�
J
.__inference_rescaling_2_layer_call_fn_44969377

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_2_layer_call_and_return_conditional_losses_44968646m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�

a
E__inference_reshape_layer_call_and_return_conditional_losses_44969417

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::��]
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: _
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:_
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:�
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_maskQ
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :Z
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
valueB :
����������
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:���������\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
F__inference_output_u_layer_call_and_return_conditional_losses_44969705

inputs0
matmul_readvariableop_resource:@
identity��MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������_
IdentityIdentityMatMul:product:0^NoOp*
T0*'
_output_shapes
:���������:
NoOpNoOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������@: 2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
d
H__inference_activation_layer_call_and_return_conditional_losses_44969754

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:���������@T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
L__inference_spatial_layer2_layer_call_and_return_conditional_losses_44968258

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
J
.__inference_rescaling_3_layer_call_fn_44969391

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_3_layer_call_and_return_conditional_losses_44968655m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
m
A__inference_add_layer_call_and_return_conditional_losses_44969640
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:���������@S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:���������@:���������@:UQ
+
_output_shapes
:���������@
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:���������@
"
_user_specified_name
inputs_0
�
�
I__inference_bve_layer_1_layer_call_and_return_conditional_losses_44968695

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@]
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@d
IdentityIdentityTanh:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
f
J__inference_activation_2_layer_call_and_return_conditional_losses_44969852

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :������������������@]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������@:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
Z
.__inference_concatenate_layer_call_fn_44969365
inputs_0
inputs_1
identity�
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_concatenate_layer_call_and_return_conditional_losses_44968663`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:���������:���������:QM
'
_output_shapes
:���������
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:���������
"
_user_specified_name
inputs_0
�
�
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968287
spatial_layer1_input)
spatial_layer1_44968274:@%
spatial_layer1_44968276:@)
spatial_layer2_44968280:@@%
spatial_layer2_44968282:@
identity��&spatial_layer1/StatefulPartitionedCall�&spatial_layer2/StatefulPartitionedCall�
&spatial_layer1/StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputspatial_layer1_44968274spatial_layer1_44968276*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_spatial_layer1_layer_call_and_return_conditional_losses_44968217�
activation/PartitionedCallPartitionedCall/spatial_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_44968227�
&spatial_layer2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0spatial_layer2_44968280spatial_layer2_44968282*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *U
fPRN
L__inference_spatial_layer2_layer_call_and_return_conditional_losses_44968258�
activation_1/PartitionedCallPartitionedCall/spatial_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_44968268x
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@t
NoOpNoOp'^spatial_layer1/StatefulPartitionedCall'^spatial_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:���������: : : : 2P
&spatial_layer1/StatefulPartitionedCall&spatial_layer1/StatefulPartitionedCall2P
&spatial_layer2/StatefulPartitionedCall&spatial_layer2/StatefulPartitionedCall:($
"
_user_specified_name
44968282:($
"
_user_specified_name
44968280:($
"
_user_specified_name
44968276:($
"
_user_specified_name
44968274:a ]
+
_output_shapes
:���������
.
_user_specified_namespatial_layer1_input
�
c
G__inference_rescaling_layer_call_and_return_conditional_losses_44968627

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ��U
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:���������Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_layer_call_fn_44969533	
query	
value
key
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44968812s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������@:������������������@:������������������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:(
$
"
_user_specified_name
44969529:(	$
"
_user_specified_name
44969527:($
"
_user_specified_name
44969525:($
"
_user_specified_name
44969523:($
"
_user_specified_name
44969521:($
"
_user_specified_name
44969519:($
"
_user_specified_name
44969517:($
"
_user_specified_name
44969515:YU
4
_output_shapes"
 :������������������@

_user_specified_namekey:[W
4
_output_shapes"
 :������������������@

_user_specified_namevalue:R N
+
_output_shapes
:���������@

_user_specified_namequery
�
K
/__inference_activation_2_layer_call_fn_44969847

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_44968383m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������@:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
�
A__inference_BPE_layer_call_and_return_conditional_losses_44968443
bpe_layer1_input%
bpe_layer1_44968430:@!
bpe_layer1_44968432:@%
bpe_layer2_44968436:@@!
bpe_layer2_44968438:@
identity��"bpe_layer1/StatefulPartitionedCall�"bpe_layer2/StatefulPartitionedCall�
"bpe_layer1/StatefulPartitionedCallStatefulPartitionedCallbpe_layer1_inputbpe_layer1_44968430bpe_layer1_44968432*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_bpe_layer1_layer_call_and_return_conditional_losses_44968373�
activation_2/PartitionedCallPartitionedCall+bpe_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_2_layer_call_and_return_conditional_losses_44968383�
"bpe_layer2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0bpe_layer2_44968436bpe_layer2_44968438*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_bpe_layer2_layer_call_and_return_conditional_losses_44968414�
activation_3/PartitionedCallPartitionedCall+bpe_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_3_layer_call_and_return_conditional_losses_44968424�
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@l
NoOpNoOp#^bpe_layer1/StatefulPartitionedCall#^bpe_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 2H
"bpe_layer1/StatefulPartitionedCall"bpe_layer1/StatefulPartitionedCall2H
"bpe_layer2/StatefulPartitionedCall"bpe_layer2/StatefulPartitionedCall:($
"
_user_specified_name
44968438:($
"
_user_specified_name
44968436:($
"
_user_specified_name
44968432:($
"
_user_specified_name
44968430:f b
4
_output_shapes"
 :������������������
*
_user_specified_namebpe_layer1_input
�	
�
&__inference_BPE_layer_call_fn_44968469
bpe_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallbpe_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :������������������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *J
fERC
A__inference_BPE_layer_call_and_return_conditional_losses_44968443|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:������������������: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44968465:($
"
_user_specified_name
44968463:($
"
_user_specified_name
44968461:($
"
_user_specified_name
44968459:f b
4
_output_shapes"
 :������������������
*
_user_specified_namebpe_layer1_input
�
�
C__inference_dense_layer_call_and_return_conditional_losses_44968867

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:���������@[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:���������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
F
*__inference_reshape_layer_call_fn_44969404

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_44968721d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_rescaling_3_layer_call_and_return_conditional_losses_44969399

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ��b
mulMulinputsCast/x:output:0*
T0*4
_output_shapes"
 :������������������g
addAddV2mul:z:0Cast_1/x:output:0*
T0*4
_output_shapes"
 :������������������\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
e
I__inference_rescaling_2_layer_call_and_return_conditional_losses_44968646

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������Q
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: b
mulMulinputsCast/x:output:0*
T0*4
_output_shapes"
 :������������������`
addAddV2mul:z:0
Cast_1:y:0*
T0*4
_output_shapes"
 :������������������\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
f
J__inference_activation_5_layer_call_and_return_conditional_losses_44969959

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:���������@P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_activation_5_layer_call_and_return_conditional_losses_44968540

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:���������@P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������@:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
7__inference_multi_head_attention_layer_call_fn_44969556	
query	
value
key
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8� *[
fVRT
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44968977s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:���������@:������������������@:������������������@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:(
$
"
_user_specified_name
44969552:(	$
"
_user_specified_name
44969550:($
"
_user_specified_name
44969548:($
"
_user_specified_name
44969546:($
"
_user_specified_name
44969544:($
"
_user_specified_name
44969542:($
"
_user_specified_name
44969540:($
"
_user_specified_name
44969538:YU
4
_output_shapes"
 :������������������@

_user_specified_namekey:[W
4
_output_shapes"
 :������������������@

_user_specified_namevalue:R N
+
_output_shapes
:���������@

_user_specified_namequery
�
f
J__inference_activation_2_layer_call_and_return_conditional_losses_44968383

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :������������������@]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������@:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
c
G__inference_rescaling_layer_call_and_return_conditional_losses_44969346

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ��U
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:���������Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_rescaling_1_layer_call_and_return_conditional_losses_44969359

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ��U
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:���������Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:���������O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
I
-__inference_activation_layer_call_fn_44969749

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *Q
fLRJ
H__inference_activation_layer_call_and_return_conditional_losses_44968227d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�	
�
$__inference_U_layer_call_fn_44968572
	ou1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCall	ou1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *H
fCRA
?__inference_U_layer_call_and_return_conditional_losses_44968543o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44968568:($
"
_user_specified_name
44968566:($
"
_user_specified_name
44968564:($
"
_user_specified_name
44968562:R N
'
_output_shapes
:���������@
#
_user_specified_name	ou1_input
�
�
H__inference_bpe_layer1_layer_call_and_return_conditional_losses_44969842

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :�������������������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :������������������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
e
I__inference_rescaling_2_layer_call_and_return_conditional_losses_44969386

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
���������Q
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: b
mulMulinputsCast/x:output:0*
T0*4
_output_shapes"
 :������������������`
addAddV2mul:z:0
Cast_1:y:0*
T0*4
_output_shapes"
 :������������������\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
�
A__inference_ou2_layer_call_and_return_conditional_losses_44968530

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�

+__inference_output_u_layer_call_fn_44969698

inputs
unknown:@
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8� *O
fJRH
F__inference_output_u_layer_call_and_return_conditional_losses_44968895o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:���������<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*(
_input_shapes
:���������@: 22
StatefulPartitionedCallStatefulPartitionedCall:($
"
_user_specified_name
44969694:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
f
J__inference_activation_3_layer_call_and_return_conditional_losses_44968424

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :������������������@]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :������������������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������@:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
a
E__inference_flatten_layer_call_and_return_conditional_losses_44969691

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"����@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:���������@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
u
K__inference_concatenate_1_layer_call_and_return_conditional_losses_44968707

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :�
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :������������������d
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:������������������:������������������:\X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�	
�
A__inference_ou1_layer_call_and_return_conditional_losses_44969920

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:���������@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:O K
'
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
L__inference_spatial_layer1_layer_call_and_return_conditional_losses_44968217

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:����������
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
L__inference_spatial_layer2_layer_call_and_return_conditional_losses_44969793

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:y
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*+
_output_shapes
:���������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:���������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:���������@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:���������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:���������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
�
I__inference_bve_layer_2_layer_call_and_return_conditional_losses_44968771

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity��BiasAdd/ReadVariableOp�Tensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0X
Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:_
Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       S
Tensordot/ShapeShapeinputs*
T0*
_output_shapes
::��Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2GatherV2Tensordot/Shape:output:0Tensordot/free:output:0 Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:[
Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/GatherV2_1GatherV2Tensordot/Shape:output:0Tensordot/axes:output:0"Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:Y
Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: n
Tensordot/ProdProdTensordot/GatherV2:output:0Tensordot/Const:output:0*
T0*
_output_shapes
: [
Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: t
Tensordot/Prod_1ProdTensordot/GatherV2_1:output:0Tensordot/Const_1:output:0*
T0*
_output_shapes
: W
Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:�
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :������������������@�
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:�������������������
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : �
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:�
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :������������������@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0�
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :������������������@]
TanhTanhBiasAdd:output:0*
T0*4
_output_shapes"
 :������������������@d
IdentityIdentityTanh:y:0^NoOp*
T0*4
_output_shapes"
 :������������������@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:������������������@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp24
Tensordot/ReadVariableOpTensordot/ReadVariableOp:($
"
_user_specified_name
resource:($
"
_user_specified_name
resource:\ X
4
_output_shapes"
 :������������������@
 
_user_specified_nameinputs
�
J
.__inference_rescaling_1_layer_call_fn_44969351

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *R
fMRK
I__inference_rescaling_1_layer_call_and_return_conditional_losses_44968636`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
e
I__inference_rescaling_3_layer_call_and_return_conditional_losses_44968655

inputs
identityK
Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @M
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ��b
mulMulinputsCast/x:output:0*
T0*4
_output_shapes"
 :������������������g
addAddV2mul:z:0Cast_1/x:output:0*
T0*4
_output_shapes"
 :������������������\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :������������������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :������������������:\ X
4
_output_shapes"
 :������������������
 
_user_specified_nameinputs
�
K
/__inference_activation_1_layer_call_fn_44969798

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:���������@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *S
fNRL
J__inference_activation_1_layer_call_and_return_conditional_losses_44968268d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:���������@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:���������@:S O
+
_output_shapes
:���������@
 
_user_specified_nameinputs
�
H
,__inference_rescaling_layer_call_fn_44969338

inputs
identity�
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8� *P
fKRI
G__inference_rescaling_layer_call_and_return_conditional_losses_44968627`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:���������"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:���������:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
L
	Xbc_layer?
serving_default_Xbc_layer:0������������������
;
t_input0
serving_default_t_input:0���������
L
	tbc_layer?
serving_default_tbc_layer:0������������������
L
	ubc_layer?
serving_default_ubc_layer:0������������������
;
x_input0
serving_default_x_input:0���������<
output_u0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�
layer-0
layer-1
layer-2
layer-3
layer-4
layer-5
layer-6
layer-7
	layer-8

layer-9
layer-10
layer-11
layer_with_weights-0
layer-12
layer_with_weights-1
layer-13
layer_with_weights-2
layer-14
layer_with_weights-3
layer-15
layer_with_weights-4
layer-16
layer-17
layer_with_weights-5
layer-18
layer-19
layer_with_weights-6
layer-20
layer_with_weights-7
layer-21
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
�
%	variables
&trainable_variables
'regularization_losses
(	keras_api
)__call__
**&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
�
+	variables
,trainable_variables
-regularization_losses
.	keras_api
/__call__
*0&call_and_return_all_conditional_losses"
_tf_keras_layer
�
1	variables
2trainable_variables
3regularization_losses
4	keras_api
5__call__
*6&call_and_return_all_conditional_losses"
_tf_keras_layer
�
7	variables
8trainable_variables
9regularization_losses
:	keras_api
;__call__
*<&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
�
=	variables
>trainable_variables
?regularization_losses
@	keras_api
A__call__
*B&call_and_return_all_conditional_losses"
_tf_keras_layer
�
C	variables
Dtrainable_variables
Eregularization_losses
F	keras_api
G__call__
*H&call_and_return_all_conditional_losses"
_tf_keras_layer
�
I	variables
Jtrainable_variables
Kregularization_losses
L	keras_api
M__call__
*N&call_and_return_all_conditional_losses

Okernel
Pbias"
_tf_keras_layer
�
Qlayer_with_weights-0
Qlayer-0
Rlayer-1
Slayer_with_weights-1
Slayer-2
Tlayer-3
U	variables
Vtrainable_variables
Wregularization_losses
X	keras_api
Y__call__
*Z&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
[layer_with_weights-0
[layer-0
\layer-1
]layer_with_weights-1
]layer-2
^layer-3
_	variables
`trainable_variables
aregularization_losses
b	keras_api
c__call__
*d&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias"
_tf_keras_layer
�
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses
s_query_dense
t
_key_dense
u_value_dense
v_softmax
w_dropout_layer
x_output_dense"
_tf_keras_layer
�
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
}__call__
*~&call_and_return_all_conditional_losses"
_tf_keras_layer
�
	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�layer_with_weights-0
�layer-0
�layer-1
�layer_with_weights-1
�layer-2
�layer-3
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_sequential
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel"
_tf_keras_layer
�
O0
P1
�2
�3
�4
�5
�6
�7
�8
�9
k10
l11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26"
trackable_list_wrapper
�
O0
P1
�2
�3
�4
�5
�6
�7
�8
�9
k10
l11
�12
�13
�14
�15
�16
�17
�18
�19
�20
�21
�22
�23
�24
�25
�26"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
trainable_variables
regularization_losses
__call__
_default_save_signature
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
(__inference_model_layer_call_fn_44969078
(__inference_model_layer_call_fn_44969141�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
C__inference_model_layer_call_and_return_conditional_losses_44968900
C__inference_model_layer_call_and_return_conditional_losses_44969015�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�B�
#__inference__wrapped_model_44968185x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"�
���
FullArgSpec
args�

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
-
�serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
,__inference_rescaling_layer_call_fn_44969338�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
G__inference_rescaling_layer_call_and_return_conditional_losses_44969346�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
%	variables
&trainable_variables
'regularization_losses
)__call__
**&call_and_return_all_conditional_losses
&*"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_rescaling_1_layer_call_fn_44969351�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_rescaling_1_layer_call_and_return_conditional_losses_44969359�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
+	variables
,trainable_variables
-regularization_losses
/__call__
*0&call_and_return_all_conditional_losses
&0"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_concatenate_layer_call_fn_44969365�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_concatenate_layer_call_and_return_conditional_losses_44969372�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
1	variables
2trainable_variables
3regularization_losses
5__call__
*6&call_and_return_all_conditional_losses
&6"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_rescaling_2_layer_call_fn_44969377�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_rescaling_2_layer_call_and_return_conditional_losses_44969386�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
7	variables
8trainable_variables
9regularization_losses
;__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_rescaling_3_layer_call_fn_44969391�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_rescaling_3_layer_call_and_return_conditional_losses_44969399�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
=	variables
>trainable_variables
?regularization_losses
A__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_reshape_layer_call_fn_44969404�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_reshape_layer_call_and_return_conditional_losses_44969417�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
C	variables
Dtrainable_variables
Eregularization_losses
G__call__
*H&call_and_return_all_conditional_losses
&H"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
0__inference_concatenate_1_layer_call_fn_44969423�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
K__inference_concatenate_1_layer_call_and_return_conditional_losses_44969430�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
.
O0
P1"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
I	variables
Jtrainable_variables
Kregularization_losses
M__call__
*N&call_and_return_all_conditional_losses
&N"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_bve_layer_1_layer_call_fn_44969439�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_bve_layer_1_layer_call_and_return_conditional_losses_44969470�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"@2bve_layer_1/kernel
:@2bve_layer_1/bias
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
U	variables
Vtrainable_variables
Wregularization_losses
Y__call__
*Z&call_and_return_all_conditional_losses
&Z"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
9__inference_spatial_transformation_layer_call_fn_44968300
9__inference_spatial_transformation_layer_call_fn_44968313�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968271
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968287�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
&__inference_BPE_layer_call_fn_44968456
&__inference_BPE_layer_call_fn_44968469�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
A__inference_BPE_layer_call_and_return_conditional_losses_44968427
A__inference_BPE_layer_call_and_return_conditional_losses_44968443�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
.
k0
l1"
trackable_list_wrapper
.
k0
l1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
.__inference_bve_layer_2_layer_call_fn_44969479�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
I__inference_bve_layer_2_layer_call_and_return_conditional_losses_44969510�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
$:"@@2bve_layer_2/kernel
:@2bve_layer_2/bias
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
`
�0
�1
�2
�3
�4
�5
�6
�7"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
7__inference_multi_head_attention_layer_call_fn_44969533
7__inference_multi_head_attention_layer_call_fn_44969556�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44969592
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44969628�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 #
defaults�

 

 
p 
p 
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�_random_generator"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�partial_output_shape
�full_output_shape
�kernel
	�bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
y	variables
ztrainable_variables
{regularization_losses
}__call__
*~&call_and_return_all_conditional_losses
&~"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_add_layer_call_fn_44969634�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_add_layer_call_and_return_conditional_losses_44969640�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
(__inference_dense_layer_call_fn_44969649�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
C__inference_dense_layer_call_and_return_conditional_losses_44969680�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
:@@2dense/kernel
:@2
dense/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
*__inference_flatten_layer_call_fn_44969685�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
E__inference_flatten_layer_call_and_return_conditional_losses_44969691�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses
�kernel
	�bias"
_tf_keras_layer
�
�	variables
�trainable_variables
�regularization_losses
�	keras_api
�__call__
+�&call_and_return_all_conditional_losses"
_tf_keras_layer
@
�0
�1
�2
�3"
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_0
�trace_12�
$__inference_U_layer_call_fn_44968572
$__inference_U_layer_call_fn_44968585�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
�
�trace_0
�trace_12�
?__inference_U_layer_call_and_return_conditional_losses_44968543
?__inference_U_layer_call_and_return_conditional_losses_44968559�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0z�trace_1
(
�0"
trackable_list_wrapper
(
�0"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
+__inference_output_u_layer_call_fn_44969698�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
F__inference_output_u_layer_call_and_return_conditional_losses_44969705�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
!:@2output_u/kernel
':%@2spatial_layer1/kernel
!:@2spatial_layer1/bias
':%@@2spatial_layer2/kernel
!:@2spatial_layer2/bias
#:!@2bpe_layer1/kernel
:@2bpe_layer1/bias
#:!@@2bpe_layer2/kernel
:@2bpe_layer2/bias
7:5@@2!multi_head_attention/query/kernel
1:/@2multi_head_attention/query/bias
5:3@@2multi_head_attention/key/kernel
/:-@2multi_head_attention/key/bias
7:5@@2!multi_head_attention/value/kernel
1:/@2multi_head_attention/value/bias
B:@@@2,multi_head_attention/attention_output/kernel
8:6@2*multi_head_attention/attention_output/bias
:@@2
ou1/kernel
:@2ou1/bias
:@@2
ou2/kernel
:@2ou2/bias
 "
trackable_list_wrapper
�
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
15
16
17
18
19
20
21"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
(__inference_model_layer_call_fn_44969078x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
(__inference_model_layer_call_fn_44969141x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_layer_call_and_return_conditional_losses_44968900x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_model_layer_call_and_return_conditional_losses_44969015x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_signature_wrapper_44969333	Xbc_layert_input	tbc_layer	ubc_layerx_input"�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 N

kwonlyargs@�=
j	Xbc_layer
	jt_input
j	tbc_layer
j	ubc_layer
	jx_input
kwonlydefaults
 
annotations� *
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
�B�
,__inference_rescaling_layer_call_fn_44969338inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
G__inference_rescaling_layer_call_and_return_conditional_losses_44969346inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_rescaling_1_layer_call_fn_44969351inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_rescaling_1_layer_call_and_return_conditional_losses_44969359inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_concatenate_layer_call_fn_44969365inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_concatenate_layer_call_and_return_conditional_losses_44969372inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_rescaling_2_layer_call_fn_44969377inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_rescaling_2_layer_call_and_return_conditional_losses_44969386inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_rescaling_3_layer_call_fn_44969391inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_rescaling_3_layer_call_and_return_conditional_losses_44969399inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_reshape_layer_call_fn_44969404inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_reshape_layer_call_and_return_conditional_losses_44969417inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
0__inference_concatenate_1_layer_call_fn_44969423inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
K__inference_concatenate_1_layer_call_and_return_conditional_losses_44969430inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_bve_layer_1_layer_call_fn_44969439inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_bve_layer_1_layer_call_and_return_conditional_losses_44969470inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_spatial_layer1_layer_call_fn_44969714�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_spatial_layer1_layer_call_and_return_conditional_losses_44969744�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_activation_layer_call_fn_44969749�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_activation_layer_call_and_return_conditional_losses_44969754�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
1__inference_spatial_layer2_layer_call_fn_44969763�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
L__inference_spatial_layer2_layer_call_and_return_conditional_losses_44969793�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_activation_1_layer_call_fn_44969798�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_activation_1_layer_call_and_return_conditional_losses_44969803�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
<
Q0
R1
S2
T3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
9__inference_spatial_transformation_layer_call_fn_44968300spatial_layer1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
9__inference_spatial_transformation_layer_call_fn_44968313spatial_layer1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968271spatial_layer1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968287spatial_layer1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_bpe_layer1_layer_call_fn_44969812�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_bpe_layer1_layer_call_and_return_conditional_losses_44969842�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_activation_2_layer_call_fn_44969847�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_activation_2_layer_call_and_return_conditional_losses_44969852�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
-__inference_bpe_layer2_layer_call_fn_44969861�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
H__inference_bpe_layer2_layer_call_and_return_conditional_losses_44969891�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_activation_3_layer_call_fn_44969896�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_activation_3_layer_call_and_return_conditional_losses_44969901�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
<
[0
\1
]2
^3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_BPE_layer_call_fn_44968456bpe_layer1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
&__inference_BPE_layer_call_fn_44968469bpe_layer1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_BPE_layer_call_and_return_conditional_losses_44968427bpe_layer1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_BPE_layer_call_and_return_conditional_losses_44968443bpe_layer1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
.__inference_bve_layer_2_layer_call_fn_44969479inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
I__inference_bve_layer_2_layer_call_and_return_conditional_losses_44969510inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
J
s0
t1
u2
v3
w4
x5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
7__inference_multi_head_attention_layer_call_fn_44969533queryvaluekey"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
7__inference_multi_head_attention_layer_call_fn_44969556queryvaluekey"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44969592queryvaluekey"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44969628queryvaluekey"�
���
FullArgSpecp
argsh�e
jquery
jvalue
jkey
jattention_mask
jreturn_attention_scores

jtraining
juse_causal_mask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�
jinputs
jmask
varargs
 
varkw
 
defaults�

 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec!
args�
jinputs

jtraining
varargs
 
varkw
 
defaults�
p 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
"
_generic_user_object
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2��
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
&__inference_add_layer_call_fn_44969634inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_add_layer_call_and_return_conditional_losses_44969640inputs_0inputs_1"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
(__inference_dense_layer_call_fn_44969649inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
C__inference_dense_layer_call_and_return_conditional_losses_44969680inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
*__inference_flatten_layer_call_fn_44969685inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
E__inference_flatten_layer_call_and_return_conditional_losses_44969691inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_ou1_layer_call_fn_44969910�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_ou1_layer_call_and_return_conditional_losses_44969920�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_activation_4_layer_call_fn_44969925�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_activation_4_layer_call_and_return_conditional_losses_44969930�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
0
�0
�1"
trackable_list_wrapper
0
�0
�1"
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
&__inference_ou2_layer_call_fn_44969939�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
A__inference_ou2_layer_call_and_return_conditional_losses_44969949�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
�
�non_trainable_variables
�layers
�metrics
 �layer_regularization_losses
�layer_metrics
�	variables
�trainable_variables
�regularization_losses
�__call__
+�&call_and_return_all_conditional_losses
'�"call_and_return_conditional_losses"
_generic_user_object
�
�trace_02�
/__inference_activation_5_layer_call_fn_44969954�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
�
�trace_02�
J__inference_activation_5_layer_call_and_return_conditional_losses_44969959�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 z�trace_0
 "
trackable_list_wrapper
@
�0
�1
�2
�3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�B�
$__inference_U_layer_call_fn_44968572	ou1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
$__inference_U_layer_call_fn_44968585	ou1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_U_layer_call_and_return_conditional_losses_44968543	ou1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
?__inference_U_layer_call_and_return_conditional_losses_44968559	ou1_input"�
���
FullArgSpec)
args!�
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
+__inference_output_u_layer_call_fn_44969698inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
F__inference_output_u_layer_call_and_return_conditional_losses_44969705inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_spatial_layer1_layer_call_fn_44969714inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_spatial_layer1_layer_call_and_return_conditional_losses_44969744inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
-__inference_activation_layer_call_fn_44969749inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_activation_layer_call_and_return_conditional_losses_44969754inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
1__inference_spatial_layer2_layer_call_fn_44969763inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
L__inference_spatial_layer2_layer_call_and_return_conditional_losses_44969793inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_activation_1_layer_call_fn_44969798inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_activation_1_layer_call_and_return_conditional_losses_44969803inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
-__inference_bpe_layer1_layer_call_fn_44969812inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_bpe_layer1_layer_call_and_return_conditional_losses_44969842inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_activation_2_layer_call_fn_44969847inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_activation_2_layer_call_and_return_conditional_losses_44969852inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
-__inference_bpe_layer2_layer_call_fn_44969861inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
H__inference_bpe_layer2_layer_call_and_return_conditional_losses_44969891inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_activation_3_layer_call_fn_44969896inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_activation_3_layer_call_and_return_conditional_losses_44969901inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
&__inference_ou1_layer_call_fn_44969910inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_ou1_layer_call_and_return_conditional_losses_44969920inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_activation_4_layer_call_fn_44969925inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_activation_4_layer_call_and_return_conditional_losses_44969930inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
&__inference_ou2_layer_call_fn_44969939inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
A__inference_ou2_layer_call_and_return_conditional_losses_44969949inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
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
�B�
/__inference_activation_5_layer_call_fn_44969954inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�B�
J__inference_activation_5_layer_call_and_return_conditional_losses_44969959inputs"�
���
FullArgSpec
args�

jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
A__inference_BPE_layer_call_and_return_conditional_losses_44968427�����N�K
D�A
7�4
bpe_layer1_input������������������
p

 
� "9�6
/�,
tensor_0������������������@
� �
A__inference_BPE_layer_call_and_return_conditional_losses_44968443�����N�K
D�A
7�4
bpe_layer1_input������������������
p 

 
� "9�6
/�,
tensor_0������������������@
� �
&__inference_BPE_layer_call_fn_44968456�����N�K
D�A
7�4
bpe_layer1_input������������������
p

 
� ".�+
unknown������������������@�
&__inference_BPE_layer_call_fn_44968469�����N�K
D�A
7�4
bpe_layer1_input������������������
p 

 
� ".�+
unknown������������������@�
?__inference_U_layer_call_and_return_conditional_losses_44968543t����:�7
0�-
#� 
	ou1_input���������@
p

 
� ",�)
"�
tensor_0���������@
� �
?__inference_U_layer_call_and_return_conditional_losses_44968559t����:�7
0�-
#� 
	ou1_input���������@
p 

 
� ",�)
"�
tensor_0���������@
� �
$__inference_U_layer_call_fn_44968572i����:�7
0�-
#� 
	ou1_input���������@
p

 
� "!�
unknown���������@�
$__inference_U_layer_call_fn_44968585i����:�7
0�-
#� 
	ou1_input���������@
p 

 
� "!�
unknown���������@�
#__inference__wrapped_model_44968185�2OP��������kl������������������
���
���
!�
x_input���������
!�
t_input���������
0�-
	Xbc_layer������������������
0�-
	tbc_layer������������������
0�-
	ubc_layer������������������
� "3�0
.
output_u"�
output_u����������
J__inference_activation_1_layer_call_and_return_conditional_losses_44969803g3�0
)�&
$�!
inputs���������@
� "0�-
&�#
tensor_0���������@
� �
/__inference_activation_1_layer_call_fn_44969798\3�0
)�&
$�!
inputs���������@
� "%�"
unknown���������@�
J__inference_activation_2_layer_call_and_return_conditional_losses_44969852y<�9
2�/
-�*
inputs������������������@
� "9�6
/�,
tensor_0������������������@
� �
/__inference_activation_2_layer_call_fn_44969847n<�9
2�/
-�*
inputs������������������@
� ".�+
unknown������������������@�
J__inference_activation_3_layer_call_and_return_conditional_losses_44969901y<�9
2�/
-�*
inputs������������������@
� "9�6
/�,
tensor_0������������������@
� �
/__inference_activation_3_layer_call_fn_44969896n<�9
2�/
-�*
inputs������������������@
� ".�+
unknown������������������@�
J__inference_activation_4_layer_call_and_return_conditional_losses_44969930_/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
/__inference_activation_4_layer_call_fn_44969925T/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
J__inference_activation_5_layer_call_and_return_conditional_losses_44969959_/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
/__inference_activation_5_layer_call_fn_44969954T/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
H__inference_activation_layer_call_and_return_conditional_losses_44969754g3�0
)�&
$�!
inputs���������@
� "0�-
&�#
tensor_0���������@
� �
-__inference_activation_layer_call_fn_44969749\3�0
)�&
$�!
inputs���������@
� "%�"
unknown���������@�
A__inference_add_layer_call_and_return_conditional_losses_44969640�b�_
X�U
S�P
&�#
inputs_0���������@
&�#
inputs_1���������@
� "0�-
&�#
tensor_0���������@
� �
&__inference_add_layer_call_fn_44969634�b�_
X�U
S�P
&�#
inputs_0���������@
&�#
inputs_1���������@
� "%�"
unknown���������@�
H__inference_bpe_layer1_layer_call_and_return_conditional_losses_44969842��<�9
2�/
-�*
inputs������������������
� "9�6
/�,
tensor_0������������������@
� �
-__inference_bpe_layer1_layer_call_fn_44969812t��<�9
2�/
-�*
inputs������������������
� ".�+
unknown������������������@�
H__inference_bpe_layer2_layer_call_and_return_conditional_losses_44969891��<�9
2�/
-�*
inputs������������������@
� "9�6
/�,
tensor_0������������������@
� �
-__inference_bpe_layer2_layer_call_fn_44969861t��<�9
2�/
-�*
inputs������������������@
� ".�+
unknown������������������@�
I__inference_bve_layer_1_layer_call_and_return_conditional_losses_44969470}OP<�9
2�/
-�*
inputs������������������
� "9�6
/�,
tensor_0������������������@
� �
.__inference_bve_layer_1_layer_call_fn_44969439rOP<�9
2�/
-�*
inputs������������������
� ".�+
unknown������������������@�
I__inference_bve_layer_2_layer_call_and_return_conditional_losses_44969510}kl<�9
2�/
-�*
inputs������������������@
� "9�6
/�,
tensor_0������������������@
� �
.__inference_bve_layer_2_layer_call_fn_44969479rkl<�9
2�/
-�*
inputs������������������@
� ".�+
unknown������������������@�
K__inference_concatenate_1_layer_call_and_return_conditional_losses_44969430�t�q
j�g
e�b
/�,
inputs_0������������������
/�,
inputs_1������������������
� "9�6
/�,
tensor_0������������������
� �
0__inference_concatenate_1_layer_call_fn_44969423�t�q
j�g
e�b
/�,
inputs_0������������������
/�,
inputs_1������������������
� ".�+
unknown�������������������
I__inference_concatenate_layer_call_and_return_conditional_losses_44969372�Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� ",�)
"�
tensor_0���������
� �
.__inference_concatenate_layer_call_fn_44969365Z�W
P�M
K�H
"�
inputs_0���������
"�
inputs_1���������
� "!�
unknown����������
C__inference_dense_layer_call_and_return_conditional_losses_44969680m��3�0
)�&
$�!
inputs���������@
� "0�-
&�#
tensor_0���������@
� �
(__inference_dense_layer_call_fn_44969649b��3�0
)�&
$�!
inputs���������@
� "%�"
unknown���������@�
E__inference_flatten_layer_call_and_return_conditional_losses_44969691c3�0
)�&
$�!
inputs���������@
� ",�)
"�
tensor_0���������@
� �
*__inference_flatten_layer_call_fn_44969685X3�0
)�&
$�!
inputs���������@
� "!�
unknown���������@�
C__inference_model_layer_call_and_return_conditional_losses_44968900�2OP��������kl������������������
���
���
!�
x_input���������
!�
t_input���������
0�-
	Xbc_layer������������������
0�-
	tbc_layer������������������
0�-
	ubc_layer������������������
p

 
� ",�)
"�
tensor_0���������
� �
C__inference_model_layer_call_and_return_conditional_losses_44969015�2OP��������kl������������������
���
���
!�
x_input���������
!�
t_input���������
0�-
	Xbc_layer������������������
0�-
	tbc_layer������������������
0�-
	ubc_layer������������������
p 

 
� ",�)
"�
tensor_0���������
� �
(__inference_model_layer_call_fn_44969078�2OP��������kl������������������
���
���
!�
x_input���������
!�
t_input���������
0�-
	Xbc_layer������������������
0�-
	tbc_layer������������������
0�-
	ubc_layer������������������
p

 
� "!�
unknown����������
(__inference_model_layer_call_fn_44969141�2OP��������kl������������������
���
���
!�
x_input���������
!�
t_input���������
0�-
	Xbc_layer������������������
0�-
	tbc_layer������������������
0�-
	ubc_layer������������������
p 

 
� "!�
unknown����������
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44969592������������
���
#� 
query���������@
,�)
value������������������@
*�'
key������������������@

 
p 
p
p 
� "0�-
&�#
tensor_0���������@
� �
R__inference_multi_head_attention_layer_call_and_return_conditional_losses_44969628������������
���
#� 
query���������@
,�)
value������������������@
*�'
key������������������@

 
p 
p 
p 
� "0�-
&�#
tensor_0���������@
� �
7__inference_multi_head_attention_layer_call_fn_44969533������������
���
#� 
query���������@
,�)
value������������������@
*�'
key������������������@

 
p 
p
p 
� "%�"
unknown���������@�
7__inference_multi_head_attention_layer_call_fn_44969556������������
���
#� 
query���������@
,�)
value������������������@
*�'
key������������������@

 
p 
p 
p 
� "%�"
unknown���������@�
A__inference_ou1_layer_call_and_return_conditional_losses_44969920e��/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
&__inference_ou1_layer_call_fn_44969910Z��/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
A__inference_ou2_layer_call_and_return_conditional_losses_44969949e��/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������@
� �
&__inference_ou2_layer_call_fn_44969939Z��/�,
%�"
 �
inputs���������@
� "!�
unknown���������@�
F__inference_output_u_layer_call_and_return_conditional_losses_44969705c�/�,
%�"
 �
inputs���������@
� ",�)
"�
tensor_0���������
� �
+__inference_output_u_layer_call_fn_44969698X�/�,
%�"
 �
inputs���������@
� "!�
unknown����������
I__inference_rescaling_1_layer_call_and_return_conditional_losses_44969359_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
.__inference_rescaling_1_layer_call_fn_44969351T/�,
%�"
 �
inputs���������
� "!�
unknown����������
I__inference_rescaling_2_layer_call_and_return_conditional_losses_44969386y<�9
2�/
-�*
inputs������������������
� "9�6
/�,
tensor_0������������������
� �
.__inference_rescaling_2_layer_call_fn_44969377n<�9
2�/
-�*
inputs������������������
� ".�+
unknown�������������������
I__inference_rescaling_3_layer_call_and_return_conditional_losses_44969399y<�9
2�/
-�*
inputs������������������
� "9�6
/�,
tensor_0������������������
� �
.__inference_rescaling_3_layer_call_fn_44969391n<�9
2�/
-�*
inputs������������������
� ".�+
unknown�������������������
G__inference_rescaling_layer_call_and_return_conditional_losses_44969346_/�,
%�"
 �
inputs���������
� ",�)
"�
tensor_0���������
� �
,__inference_rescaling_layer_call_fn_44969338T/�,
%�"
 �
inputs���������
� "!�
unknown����������
E__inference_reshape_layer_call_and_return_conditional_losses_44969417c/�,
%�"
 �
inputs���������
� "0�-
&�#
tensor_0���������
� �
*__inference_reshape_layer_call_fn_44969404X/�,
%�"
 �
inputs���������
� "%�"
unknown����������
&__inference_signature_wrapper_44969333�2OP��������kl������������������
� 
���
=
	Xbc_layer0�-
	xbc_layer������������������
,
t_input!�
t_input���������
=
	tbc_layer0�-
	tbc_layer������������������
=
	ubc_layer0�-
	ubc_layer������������������
,
x_input!�
x_input���������"3�0
.
output_u"�
output_u����������
L__inference_spatial_layer1_layer_call_and_return_conditional_losses_44969744m��3�0
)�&
$�!
inputs���������
� "0�-
&�#
tensor_0���������@
� �
1__inference_spatial_layer1_layer_call_fn_44969714b��3�0
)�&
$�!
inputs���������
� "%�"
unknown���������@�
L__inference_spatial_layer2_layer_call_and_return_conditional_losses_44969793m��3�0
)�&
$�!
inputs���������@
� "0�-
&�#
tensor_0���������@
� �
1__inference_spatial_layer2_layer_call_fn_44969763b��3�0
)�&
$�!
inputs���������@
� "%�"
unknown���������@�
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968271�����I�F
?�<
2�/
spatial_layer1_input���������
p

 
� "0�-
&�#
tensor_0���������@
� �
T__inference_spatial_transformation_layer_call_and_return_conditional_losses_44968287�����I�F
?�<
2�/
spatial_layer1_input���������
p 

 
� "0�-
&�#
tensor_0���������@
� �
9__inference_spatial_transformation_layer_call_fn_44968300|����I�F
?�<
2�/
spatial_layer1_input���������
p

 
� "%�"
unknown���������@�
9__inference_spatial_transformation_layer_call_fn_44968313|����I�F
?�<
2�/
spatial_layer1_input���������
p 

 
� "%�"
unknown���������@