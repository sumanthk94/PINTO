Щ(
§б
D
AddV2
x"T
y"T
z"T"
Ttype:
2	
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 

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
resource
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
Ў
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
9
	IdentityN

input2T
output2T"
T
list(type)(0
u
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
2	

MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( 
?
Mul
x"T
y"T
z"T"
Ttype:
2	
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

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
dtypetype
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
d
Shape

input"T&
output"out_typeэout_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
9
Softmax
logits"T
softmax"T"
Ttype:
2
С
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
executor_typestring Ј
@
StaticRegexFullMatch	
input

output
"
patternstring
ї
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
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
А
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.15.02v2.15.0-2-g0b15fdfcb3f8фМ 

op2/biasVarHandleOp*
_output_shapes
: *

debug_name	op2/bias/*
dtype0*
shape:@*
shared_name
op2/bias
a
op2/bias/Read/ReadVariableOpReadVariableOpop2/bias*
_output_shapes
:@*
dtype0


op2/kernelVarHandleOp*
_output_shapes
: *

debug_nameop2/kernel/*
dtype0*
shape
:@@*
shared_name
op2/kernel
i
op2/kernel/Read/ReadVariableOpReadVariableOp
op2/kernel*
_output_shapes

:@@*
dtype0

op1/biasVarHandleOp*
_output_shapes
: *

debug_name	op1/bias/*
dtype0*
shape:@*
shared_name
op1/bias
a
op1/bias/Read/ReadVariableOpReadVariableOpop1/bias*
_output_shapes
:@*
dtype0


op1/kernelVarHandleOp*
_output_shapes
: *

debug_nameop1/kernel/*
dtype0*
shape
:@@*
shared_name
op1/kernel
i
op1/kernel/Read/ReadVariableOpReadVariableOp
op1/kernel*
_output_shapes

:@@*
dtype0

ov2/biasVarHandleOp*
_output_shapes
: *

debug_name	ov2/bias/*
dtype0*
shape:@*
shared_name
ov2/bias
a
ov2/bias/Read/ReadVariableOpReadVariableOpov2/bias*
_output_shapes
:@*
dtype0


ov2/kernelVarHandleOp*
_output_shapes
: *

debug_nameov2/kernel/*
dtype0*
shape
:@@*
shared_name
ov2/kernel
i
ov2/kernel/Read/ReadVariableOpReadVariableOp
ov2/kernel*
_output_shapes

:@@*
dtype0

ov1/biasVarHandleOp*
_output_shapes
: *

debug_name	ov1/bias/*
dtype0*
shape:@*
shared_name
ov1/bias
a
ov1/bias/Read/ReadVariableOpReadVariableOpov1/bias*
_output_shapes
:@*
dtype0


ov1/kernelVarHandleOp*
_output_shapes
: *

debug_nameov1/kernel/*
dtype0*
shape
:@@*
shared_name
ov1/kernel
i
ov1/kernel/Read/ReadVariableOpReadVariableOp
ov1/kernel*
_output_shapes

:@@*
dtype0

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


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

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


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
щ
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *;

debug_name-+multi_head_attention/attention_output/bias/*
dtype0*
shape:@*;
shared_name,*multi_head_attention/attention_output/bias
Ѕ
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:@*
dtype0
ї
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *=

debug_name/-multi_head_attention/attention_output/kernel/*
dtype0*
shape:@@*=
shared_name.,multi_head_attention/attention_output/kernel
Б
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:@@*
dtype0
Ь
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/value/bias/*
dtype0*
shape
:@*0
shared_name!multi_head_attention/value/bias

3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:@*
dtype0
ж
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/value/kernel/*
dtype0*
shape:@@*2
shared_name#!multi_head_attention/value/kernel

5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:@@*
dtype0
Ц
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *.

debug_name multi_head_attention/key/bias/*
dtype0*
shape
:@*.
shared_namemulti_head_attention/key/bias

1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:@*
dtype0
а
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/key/kernel/*
dtype0*
shape:@@*0
shared_name!multi_head_attention/key/kernel

3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:@@*
dtype0
Ь
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/query/bias/*
dtype0*
shape
:@*0
shared_name!multi_head_attention/query/bias

3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:@*
dtype0
ж
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/query/kernel/*
dtype0*
shape:@@*2
shared_name#!multi_head_attention/query/kernel

5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:@@*
dtype0
Є
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
Ў
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
Є
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
Ў
spatial_layer1/kernelVarHandleOp*
_output_shapes
: *&

debug_namespatial_layer1/kernel/*
dtype0*
shape
:@*&
shared_namespatial_layer1/kernel

)spatial_layer1/kernel/Read/ReadVariableOpReadVariableOpspatial_layer1/kernel*
_output_shapes

:@*
dtype0

output_p/biasVarHandleOp*
_output_shapes
: *

debug_nameoutput_p/bias/*
dtype0*
shape:*
shared_nameoutput_p/bias
k
!output_p/bias/Read/ReadVariableOpReadVariableOpoutput_p/bias*
_output_shapes
:*
dtype0

output_p/kernelVarHandleOp*
_output_shapes
: * 

debug_nameoutput_p/kernel/*
dtype0*
shape
:@* 
shared_nameoutput_p/kernel
s
#output_p/kernel/Read/ReadVariableOpReadVariableOpoutput_p/kernel*
_output_shapes

:@*
dtype0

output_v/biasVarHandleOp*
_output_shapes
: *

debug_nameoutput_v/bias/*
dtype0*
shape:*
shared_nameoutput_v/bias
k
!output_v/bias/Read/ReadVariableOpReadVariableOpoutput_v/bias*
_output_shapes
:*
dtype0

output_v/kernelVarHandleOp*
_output_shapes
: * 

debug_nameoutput_v/kernel/*
dtype0*
shape
:@* 
shared_nameoutput_v/kernel
s
#output_v/kernel/Read/ReadVariableOpReadVariableOpoutput_v/kernel*
_output_shapes

:@*
dtype0

output_u/biasVarHandleOp*
_output_shapes
: *

debug_nameoutput_u/bias/*
dtype0*
shape:*
shared_nameoutput_u/bias
k
!output_u/bias/Read/ReadVariableOpReadVariableOpoutput_u/bias*
_output_shapes
:*
dtype0

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

dense_4/biasVarHandleOp*
_output_shapes
: *

debug_namedense_4/bias/*
dtype0*
shape:@*
shared_namedense_4/bias
i
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes
:@*
dtype0

dense_4/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_4/kernel/*
dtype0*
shape
:@@*
shared_namedense_4/kernel
q
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel*
_output_shapes

:@@*
dtype0

dense_3/biasVarHandleOp*
_output_shapes
: *

debug_namedense_3/bias/*
dtype0*
shape:@*
shared_namedense_3/bias
i
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes
:@*
dtype0

dense_3/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_3/kernel/*
dtype0*
shape
:@@*
shared_namedense_3/kernel
q
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel*
_output_shapes

:@@*
dtype0

dense_1/biasVarHandleOp*
_output_shapes
: *

debug_namedense_1/bias/*
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0

dense_1/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_1/kernel/*
dtype0*
shape
:@@*
shared_namedense_1/kernel
q
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes

:@@*
dtype0

dense_2/biasVarHandleOp*
_output_shapes
: *

debug_namedense_2/bias/*
dtype0*
shape:@*
shared_namedense_2/bias
i
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes
:@*
dtype0

dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape
:@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@*
dtype0


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

dense/kernelVarHandleOp*
_output_shapes
: *

debug_namedense/kernel/*
dtype0*
shape
:@*
shared_namedense/kernel
m
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes

:@*
dtype0

serving_default_Pbc_layerPlaceholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ
z
serving_default_T_layerPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

serving_default_Tbc_layerPlaceholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ

serving_default_Ubc_layerPlaceholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ

serving_default_Vbc_layerPlaceholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ
z
serving_default_X_layerPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

serving_default_Xbc_layerPlaceholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ
z
serving_default_Y_layerPlaceholder*'
_output_shapes
:џџџџџџџџџ*
dtype0*
shape:џџџџџџџџџ

serving_default_Ybc_layerPlaceholder*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*
dtype0*)
shape :џџџџџџџџџџџџџџџџџџ
њ

StatefulPartitionedCallStatefulPartitionedCallserving_default_Pbc_layerserving_default_T_layerserving_default_Tbc_layerserving_default_Ubc_layerserving_default_Vbc_layerserving_default_X_layerserving_default_Xbc_layerserving_default_Y_layerserving_default_Ybc_layerdense_2/kerneldense_2/biasdense/kernel
dense/biasspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/biasdense_1/kerneldense_1/biasdense_3/kerneldense_3/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biasdense_4/kerneldense_4/bias
op1/kernelop1/bias
op2/kernelop2/bias
ov1/kernelov1/bias
ov2/kernelov2/bias
ou1/kernelou1/bias
ou2/kernelou2/biasoutput_p/kerneloutput_p/biasoutput_v/kerneloutput_v/biasoutput_u/kerneloutput_u/bias*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8 *.
f)R'
%__inference_signature_wrapper_9236054

NoOpNoOp
я
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*аю
valueХюBСю BЙю
Ќ
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
layer-12
layer-13
layer-14
layer_with_weights-0
layer-15
layer_with_weights-1
layer-16
layer_with_weights-2
layer-17
layer_with_weights-3
layer-18
layer_with_weights-4
layer-19
layer_with_weights-5
layer-20
layer-21
layer_with_weights-6
layer-22
layer-23
layer_with_weights-7
layer-24
layer_with_weights-8
layer-25
layer_with_weights-9
layer-26
layer_with_weights-10
layer-27
layer_with_weights-11
layer-28
layer_with_weights-12
layer-29
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature
&
signatures*
* 
* 
* 
* 

'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses* 
* 
* 

-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses* 
* 
* 
* 

3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses* 

9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses* 

?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses* 

E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses* 
І
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias*
І
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias*
ј
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
І
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias*
І
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias*
њ
u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_query_dense
|
_key_dense
}_value_dense
~_softmax
_dropout_layer
_output_dense*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 
Ў
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias*

	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses* 

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses*

layer_with_weights-0
layer-0
 layer-1
Ёlayer_with_weights-1
Ёlayer-2
Ђlayer-3
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
І	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses*

Љlayer_with_weights-0
Љlayer-0
Њlayer-1
Ћlayer_with_weights-1
Ћlayer-2
Ќlayer-3
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses*
Ў
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses
Йkernel
	Кbias*
Ў
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses
Сkernel
	Тbias*
Ў
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses
Щkernel
	Ъbias*
к
Q0
R1
Y2
Z3
Ы4
Ь5
Э6
Ю7
k8
l9
s10
t11
Я12
а13
б14
в15
г16
д17
е18
ж19
20
21
з22
и23
й24
к25
л26
м27
н28
о29
п30
р31
с32
т33
Й34
К35
С36
Т37
Щ38
Ъ39*
к
Q0
R1
Y2
Z3
Ы4
Ь5
Э6
Ю7
k8
l9
s10
t11
Я12
а13
б14
в15
г16
д17
е18
ж19
20
21
з22
и23
й24
к25
л26
м27
н28
о29
п30
р31
с32
т33
Й34
К35
С36
Т37
Щ38
Ъ39*
* 
Е
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*

шtrace_0
щtrace_1* 

ъtrace_0
ыtrace_1* 
* 

ьserving_default* 
* 
* 
* 

эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses* 

ђtrace_0* 

ѓtrace_0* 
* 
* 
* 

єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses* 

љtrace_0* 

њtrace_0* 
* 
* 
* 

ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

Q0
R1*

Q0
R1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses*

trace_0* 

trace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

Y0
Z1*

Y0
Z1*
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses*

Ѓtrace_0* 

Єtrace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
Ў
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses
Ыkernel
	Ьbias*

Ћ	variables
Ќtrainable_variables
­regularization_losses
Ў	keras_api
Џ__call__
+А&call_and_return_all_conditional_losses* 
Ў
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
Эkernel
	Юbias*

З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses* 
$
Ы0
Ь1
Э2
Ю3*
$
Ы0
Ь1
Э2
Ю3*
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses*

Тtrace_0
Уtrace_1* 

Фtrace_0
Хtrace_1* 

k0
l1*

k0
l1*
* 

Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses*

Ыtrace_0* 

Ьtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE*

s0
t1*

s0
t1*
* 

Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses*

вtrace_0* 

гtrace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*
D
Я0
а1
б2
в3
г4
д5
е6
ж7*
D
Я0
а1
б2
в3
г4
д5
е6
ж7*
* 

дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses*

йtrace_0
кtrace_1* 

лtrace_0
мtrace_1* 
с
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses
уpartial_output_shape
фfull_output_shape
Яkernel
	аbias*
с
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses
ыpartial_output_shape
ьfull_output_shape
бkernel
	вbias*
с
э	variables
юtrainable_variables
яregularization_losses
№	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses
ѓpartial_output_shape
єfull_output_shape
гkernel
	дbias*

ѕ	variables
іtrainable_variables
їregularization_losses
ј	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses* 
Ќ
ћ	variables
ќtrainable_variables
§regularization_losses
ў	keras_api
џ__call__
+&call_and_return_all_conditional_losses
_random_generator* 
с
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
еkernel
	жbias*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

0
1*

0
1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

trace_0* 

trace_0* 
^X
VARIABLE_VALUEdense_4/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_4/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
Ў
	variables
 trainable_variables
Ёregularization_losses
Ђ	keras_api
Ѓ__call__
+Є&call_and_return_all_conditional_losses
зkernel
	иbias*

Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses* 
Ў
Ћ	variables
Ќtrainable_variables
­regularization_losses
Ў	keras_api
Џ__call__
+А&call_and_return_all_conditional_losses
йkernel
	кbias*

Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses* 
$
з0
и1
й2
к3*
$
з0
и1
й2
к3*
* 

Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*

Мtrace_0
Нtrace_1* 

Оtrace_0
Пtrace_1* 
Ў
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses
лkernel
	мbias*

Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses* 
Ў
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses
нkernel
	оbias*

в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses* 
$
л0
м1
н2
о3*
$
л0
м1
н2
о3*
* 

иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses*

нtrace_0
оtrace_1* 

пtrace_0
рtrace_1* 
Ў
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
пkernel
	рbias*

ч	variables
шtrainable_variables
щregularization_losses
ъ	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses* 
Ў
э	variables
юtrainable_variables
яregularization_losses
№	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses
сkernel
	тbias*

ѓ	variables
єtrainable_variables
ѕregularization_losses
і	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses* 
$
п0
р1
с2
т3*
$
п0
р1
с2
т3*
* 

љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses*

ўtrace_0
џtrace_1* 

trace_0
trace_1* 

Й0
К1*

Й0
К1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEoutput_u/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_u/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*

С0
Т1*

С0
Т1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEoutput_v/kernel7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_v/bias5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUE*

Щ0
Ъ1*

Щ0
Ъ1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses*

trace_0* 

trace_0* 
`Z
VARIABLE_VALUEoutput_p/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_p/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEspatial_layer1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEspatial_layer1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEspatial_layer2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEspatial_layer2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
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
KE
VARIABLE_VALUE
ov1/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEov1/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
ov2/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEov2/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
op1/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEop1/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
op2/kernel'variables/32/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEop2/bias'variables/33/.ATTRIBUTES/VARIABLE_VALUE*
* 
ъ
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
21
22
23
24
25
26
27
28
29*
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
Ы0
Ь1*

Ы0
Ь1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
Ћ	variables
Ќtrainable_variables
­regularization_losses
Џ__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses* 

Ѓtrace_0* 

Єtrace_0* 

Э0
Ю1*

Э0
Ю1*
* 

Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses*

Њtrace_0* 

Ћtrace_0* 
* 
* 
* 

Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses* 

Бtrace_0* 

Вtrace_0* 
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
* 
* 
* 
* 
* 
* 
* 
/
{0
|1
}2
~3
4
5*
* 
* 
* 
* 
* 
* 
* 

Я0
а1*

Я0
а1*
* 

Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses*
* 
* 
* 
* 

б0
в1*

б0
в1*
* 

Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses*
* 
* 
* 
* 

г0
д1*

г0
д1*
* 

Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
э	variables
юtrainable_variables
яregularization_losses
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 

Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
ѕ	variables
іtrainable_variables
їregularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 

Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
ћ	variables
ќtrainable_variables
§regularization_losses
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses* 
* 
* 
* 

е0
ж1*

е0
ж1*
* 

Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses*
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
з0
и1*

з0
и1*
* 

бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
	variables
 trainable_variables
Ёregularization_losses
Ѓ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses*

жtrace_0* 

зtrace_0* 
* 
* 
* 

иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses* 

нtrace_0* 

оtrace_0* 

й0
к1*

й0
к1*
* 

пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
Ћ	variables
Ќtrainable_variables
­regularization_losses
Џ__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses*

фtrace_0* 

хtrace_0* 
* 
* 
* 

цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses* 

ыtrace_0* 

ьtrace_0* 
* 
$
0
1
2
3*
* 
* 
* 
* 
* 
* 
* 

л0
м1*

л0
м1*
* 

эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses*

ђtrace_0* 

ѓtrace_0* 
* 
* 
* 

єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses* 

љtrace_0* 

њtrace_0* 

н0
о1*

н0
о1*
* 

ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses* 

trace_0* 

trace_0* 
* 
$
0
 1
Ё2
Ђ3*
* 
* 
* 
* 
* 
* 
* 

п0
р1*

п0
р1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ч	variables
шtrainable_variables
щregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses* 

trace_0* 

trace_0* 

с0
т1*

с0
т1*
* 

non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
э	variables
юtrainable_variables
яregularization_losses
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses*

trace_0* 

trace_0* 
* 
* 
* 

non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
ѓ	variables
єtrainable_variables
ѕregularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses* 

Ѓtrace_0* 

Єtrace_0* 
* 
$
Љ0
Њ1
Ћ2
Ќ3*
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
З
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_2/kerneldense_2/biasdense_1/kerneldense_1/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasoutput_u/kerneloutput_u/biasoutput_v/kerneloutput_v/biasoutput_p/kerneloutput_p/biasspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias
ou1/kernelou1/bias
ou2/kernelou2/bias
ov1/kernelov1/bias
ov2/kernelov2/bias
op1/kernelop1/bias
op2/kernelop2/biasConst*5
Tin.
,2**
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
GPU2*0J 8 *)
f$R"
 __inference__traced_save_9238246
В
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_2/kerneldense_2/biasdense_1/kerneldense_1/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasoutput_u/kerneloutput_u/biasoutput_v/kerneloutput_v/biasoutput_p/kerneloutput_p/biasspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias
ou1/kernelou1/bias
ou2/kernelou2/bias
ov1/kernelov1/bias
ov2/kernelov2/bias
op1/kernelop1/bias
op2/kernelop2/bias*4
Tin-
+2)*
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
GPU2*0J 8 *,
f'R%
#__inference__traced_restore_9238375ТЮ
Т
З
$__inference_internal_grad_fn_9237770
result_grads_0
result_grads_1
result_grads_2
mul_model_dense_beta
mul_model_dense_biasadd
identity

identity_1
mulMulmul_model_dense_betamul_model_dense_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
mul_1Mulmul_model_dense_betamul_model_dense_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@h
SquareSquaremul_model_dense_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:ie
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
-
_user_specified_namemodel/dense/BiasAdd:HD

_output_shapes
: 
*
_user_specified_namemodel/dense/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0


K__inference_spatial_layer1_layer_call_and_return_conditional_losses_9236624

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 20
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
:џџџџџџџџџ
 
_user_specified_nameinputs
ј	
і
E__inference_output_p_layer_call_and_return_conditional_losses_9235407

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ї

)__inference_dense_3_layer_call_fn_9236300

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_9235254|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236296:'#
!
_user_specified_name	9236294:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
	
Щ
#__inference_P_layer_call_fn_9234973
	op1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCall	op1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_P_layer_call_and_return_conditional_losses_9234947o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9234969:'#
!
_user_specified_name	9234967:'#
!
_user_specified_name	9234965:'#
!
_user_specified_name	9234963:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	op1_input
#
§
D__inference_dense_3_layer_call_and_return_conditional_losses_9235254

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@з
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9235245*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ@: : 20
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
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs


K__inference_spatial_layer2_layer_call_and_return_conditional_losses_9236681

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
"
§
D__inference_dense_4_layer_call_and_return_conditional_losses_9236517

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@a
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Х
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236508*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
	
Щ
#__inference_V_layer_call_fn_9234841
	ov1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCall	ov1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_V_layer_call_and_return_conditional_losses_9234815o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9234837:'#
!
_user_specified_name	9234835:'#
!
_user_specified_name	9234833:'#
!
_user_specified_name	9234831:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	ov1_input
	
Щ
#__inference_U_layer_call_fn_9234709
	ou1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCall	ou1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_U_layer_call_and_return_conditional_losses_9234683o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9234705:'#
!
_user_specified_name	9234703:'#
!
_user_specified_name	9234701:'#
!
_user_specified_name	9234699:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	ou1_input
Ц
ђ
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234535
spatial_layer1_input(
spatial_layer1_9234466:@$
spatial_layer1_9234468:@(
spatial_layer2_9234515:@@$
spatial_layer2_9234517:@
identityЂ&spatial_layer1/StatefulPartitionedCallЂ&spatial_layer2/StatefulPartitionedCall 
&spatial_layer1/StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputspatial_layer1_9234466spatial_layer1_9234468*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_9234465э
activation/PartitionedCallPartitionedCall/spatial_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_9234483Џ
&spatial_layer2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0spatial_layer2_9234515spatial_layer2_9234517*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_9234514ё
activation_1/PartitionedCallPartitionedCall/spatial_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_9234532x
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@t
NoOpNoOp'^spatial_layer1/StatefulPartitionedCall'^spatial_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2P
&spatial_layer1/StatefulPartitionedCall&spatial_layer1/StatefulPartitionedCall2P
&spatial_layer2/StatefulPartitionedCall&spatial_layer2/StatefulPartitionedCall:'#
!
_user_specified_name	9234517:'#
!
_user_specified_name	9234515:'#
!
_user_specified_name	9234468:'#
!
_user_specified_name	9234466:a ]
+
_output_shapes
:џџџџџџџџџ
.
_user_specified_namespatial_layer1_input
#
§
D__inference_dense_3_layer_call_and_return_conditional_losses_9236339

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@з
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236330*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ@: : 20
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
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs

Л
$__inference_internal_grad_fn_9237905
result_grads_0
result_grads_1
result_grads_2
mul_model_dense_4_beta
mul_model_dense_4_biasadd
identity

identity_1
mulMulmul_model_dense_4_betamul_model_dense_4_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@u
mul_1Mulmul_model_dense_4_betamul_model_dense_4_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@a
SquareSquaremul_model_dense_4_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:b^
+
_output_shapes
:џџџџџџџџџ@
/
_user_specified_namemodel/dense_4/BiasAdd:JF

_output_shapes
: 
,
_user_specified_namemodel/dense_4/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Ц

$__inference_internal_grad_fn_9237230
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Њ
J
.__inference_activation_5_layer_call_fn_9236834

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_9234796`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѓ	
ё
@__inference_ov1_layer_call_and_return_conditional_losses_9236792

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
#
ћ
B__inference_dense_layer_call_and_return_conditional_losses_9236195

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@з
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236186*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ: : 20
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
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
е

$__inference_internal_grad_fn_9237608
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@b
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:]Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0

i
/__inference_concatenate_1_layer_call_fn_9236106
inputs_0
inputs_1
inputs_2
identityн
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9235050m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0
д
Л
$__inference_internal_grad_fn_9237743
result_grads_0
result_grads_1
result_grads_2
mul_model_dense_2_beta
mul_model_dense_2_biasadd
identity

identity_1
mulMulmul_model_dense_2_betamul_model_dense_2_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@~
mul_1Mulmul_model_dense_2_betamul_model_dense_2_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
SquareSquaremul_model_dense_2_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:kg
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
/
_user_specified_namemodel/dense_2/BiasAdd:JF

_output_shapes
: 
,
_user_specified_namemodel/dense_2/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
м
Р
$__inference_internal_grad_fn_9238013
result_grads_0
result_grads_1
result_grads_2!
mul_model_v_activation_5_beta
mul_model_v_ov2_biasadd
identity

identity_1
mulMulmul_model_v_activation_5_betamul_model_v_ov2_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
mul_1Mulmul_model_v_activation_5_betamul_model_v_ov2_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
SquareSquaremul_model_v_ov2_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:\X
'
_output_shapes
:џџџџџџџџџ@
-
_user_specified_namemodel/V/ov2/BiasAdd:QM

_output_shapes
: 
3
_user_specified_namemodel/V/activation_5/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
ќ
e
G__inference_activation_layer_call_and_return_conditional_losses_9234483

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
mulMulbeta:output:0inputs*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@W
mul_1MulinputsSigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Л
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234474*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: `

Identity_1IdentityIdentityN:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Н.

Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9236457	
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
identityЂ#attention_output/add/ReadVariableOpЂ-attention_output/einsum/Einsum/ReadVariableOpЂkey/add/ReadVariableOpЂ key/einsum/Einsum/ReadVariableOpЂquery/add/ReadVariableOpЂ"query/einsum/Einsum/ReadVariableOpЂvalue/add/ReadVariableOpЂ"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0А
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Г
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Й
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџz
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџЉ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationacbe,aecd->abcdЈ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0е
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0­
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@Д
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:џџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : : : : : : : 2J
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
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namekey:[W
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namevalue:R N
+
_output_shapes
:џџџџџџџџџ@

_user_specified_namequery
к
g
I__inference_activation_7_layer_call_and_return_conditional_losses_9234928

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234919*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ї
ѕ
$__inference_internal_grad_fn_9237824
result_grads_0
result_grads_1
result_grads_26
2mul_model_spatial_transformation_activation_1_beta;
7mul_model_spatial_transformation_spatial_layer2_biasadd
identity

identity_1О
mulMul2mul_model_spatial_transformation_activation_1_beta7mul_model_spatial_transformation_spatial_layer2_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Џ
mul_1Mul2mul_model_spatial_transformation_activation_1_beta7mul_model_spatial_transformation_spatial_layer2_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@
SquareSquare7mul_model_spatial_transformation_spatial_layer2_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:|
+
_output_shapes
:џџџџџџџџџ@
M
_user_specified_name53model/spatial_transformation/spatial_layer2/BiasAdd:fb

_output_shapes
: 
H
_user_specified_name0.model/spatial_transformation/activation_1/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
"
§
D__inference_dense_4_layer_call_and_return_conditional_losses_9235358

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
mulMulbeta:output:0BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@a
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Х
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9235349*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: g

Identity_1IdentityIdentityN:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
ќ
e
G__inference_activation_layer_call_and_return_conditional_losses_9236642

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
mulMulbeta:output:0inputs*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@W
mul_1MulinputsSigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Л
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236633*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: `

Identity_1IdentityIdentityN:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
е

$__inference_internal_grad_fn_9237662
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@b
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:]Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Б-
ф

'__inference_model_layer_call_fn_9235694
x_layer
y_layer
t_layer
	xbc_layer
	ybc_layer
	tbc_layer
	ubc_layer
	vbc_layer
	pbc_layer
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
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

unknown_25:@@

unknown_26:@

unknown_27:@@

unknown_28:@

unknown_29:@@

unknown_30:@

unknown_31:@@

unknown_32:@

unknown_33:@

unknown_34:

unknown_35:@

unknown_36:

unknown_37:@

unknown_38:
identity

identity_1

identity_2ЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallx_layery_layert_layer	xbc_layer	ybc_layer	tbc_layer	ubc_layer	vbc_layer	pbc_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_9235446o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'0#
!
_user_specified_name	9235686:'/#
!
_user_specified_name	9235684:'.#
!
_user_specified_name	9235682:'-#
!
_user_specified_name	9235680:',#
!
_user_specified_name	9235678:'+#
!
_user_specified_name	9235676:'*#
!
_user_specified_name	9235674:')#
!
_user_specified_name	9235672:'(#
!
_user_specified_name	9235670:''#
!
_user_specified_name	9235668:'&#
!
_user_specified_name	9235666:'%#
!
_user_specified_name	9235664:'$#
!
_user_specified_name	9235662:'##
!
_user_specified_name	9235660:'"#
!
_user_specified_name	9235658:'!#
!
_user_specified_name	9235656:' #
!
_user_specified_name	9235654:'#
!
_user_specified_name	9235652:'#
!
_user_specified_name	9235650:'#
!
_user_specified_name	9235648:'#
!
_user_specified_name	9235646:'#
!
_user_specified_name	9235644:'#
!
_user_specified_name	9235642:'#
!
_user_specified_name	9235640:'#
!
_user_specified_name	9235638:'#
!
_user_specified_name	9235636:'#
!
_user_specified_name	9235634:'#
!
_user_specified_name	9235632:'#
!
_user_specified_name	9235630:'#
!
_user_specified_name	9235628:'#
!
_user_specified_name	9235626:'#
!
_user_specified_name	9235624:'#
!
_user_specified_name	9235622:'#
!
_user_specified_name	9235620:'#
!
_user_specified_name	9235618:'#
!
_user_specified_name	9235616:'#
!
_user_specified_name	9235614:'#
!
_user_specified_name	9235612:'
#
!
_user_specified_name	9235610:'	#
!
_user_specified_name	9235608:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Pbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Vbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ubc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Tbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ybc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	T_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Y_layer:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	X_layer
м
Р
$__inference_internal_grad_fn_9237932
result_grads_0
result_grads_1
result_grads_2!
mul_model_p_activation_6_beta
mul_model_p_op1_biasadd
identity

identity_1
mulMulmul_model_p_activation_6_betamul_model_p_op1_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
mul_1Mulmul_model_p_activation_6_betamul_model_p_op1_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
SquareSquaremul_model_p_op1_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:\X
'
_output_shapes
:џџџџџџџџџ@
-
_user_specified_namemodel/P/op1/BiasAdd:QM

_output_shapes
: 
3
_user_specified_namemodel/P/activation_6/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0


>__inference_P_layer_call_and_return_conditional_losses_9234947
	op1_input
op1_9234934:@@
op1_9234936:@
op2_9234940:@@
op2_9234942:@
identityЂop1/StatefulPartitionedCallЂop2/StatefulPartitionedCallх
op1/StatefulPartitionedCallStatefulPartitionedCall	op1_inputop1_9234934op1_9234936*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_op1_layer_call_and_return_conditional_losses_9234881т
activation_6/PartitionedCallPartitionedCall$op1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_9234899
op2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0op2_9234940op2_9234942*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_op2_layer_call_and_return_conditional_losses_9234910т
activation_7/PartitionedCallPartitionedCall$op2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_9234928t
IdentityIdentity%activation_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@^
NoOpNoOp^op1/StatefulPartitionedCall^op2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 2:
op1/StatefulPartitionedCallop1/StatefulPartitionedCall2:
op2/StatefulPartitionedCallop2/StatefulPartitionedCall:'#
!
_user_specified_name	9234942:'#
!
_user_specified_name	9234940:'#
!
_user_specified_name	9234936:'#
!
_user_specified_name	9234934:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	op1_input


$__inference_internal_grad_fn_9237365
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1g
mulMulmul_beta
mul_inputs^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_1Mulmul_beta
mul_inputs*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@R
SquareSquare
mul_inputs*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:SO
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Њ
J
.__inference_activation_2_layer_call_fn_9236723

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_9234635`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ы

%__inference_ou2_layer_call_fn_9236745

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ou2_layer_call_and_return_conditional_losses_9234646o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236741:'#
!
_user_specified_name	9236739:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


>__inference_U_layer_call_and_return_conditional_losses_9234667
	ou1_input
ou1_9234618:@@
ou1_9234620:@
ou2_9234647:@@
ou2_9234649:@
identityЂou1/StatefulPartitionedCallЂou2/StatefulPartitionedCallх
ou1/StatefulPartitionedCallStatefulPartitionedCall	ou1_inputou1_9234618ou1_9234620*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ou1_layer_call_and_return_conditional_losses_9234617т
activation_2/PartitionedCallPartitionedCall$ou1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_9234635
ou2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0ou2_9234647ou2_9234649*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ou2_layer_call_and_return_conditional_losses_9234646т
activation_3/PartitionedCallPartitionedCall$ou2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_9234664t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@^
NoOpNoOp^ou1/StatefulPartitionedCall^ou2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 2:
ou1/StatefulPartitionedCallou1/StatefulPartitionedCall2:
ou2/StatefulPartitionedCallou2/StatefulPartitionedCall:'#
!
_user_specified_name	9234649:'#
!
_user_specified_name	9234647:'#
!
_user_specified_name	9234620:'#
!
_user_specified_name	9234618:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	ou1_input
д
Л
$__inference_internal_grad_fn_9237851
result_grads_0
result_grads_1
result_grads_2
mul_model_dense_1_beta
mul_model_dense_1_biasadd
identity

identity_1
mulMulmul_model_dense_1_betamul_model_dense_1_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@~
mul_1Mulmul_model_dense_1_betamul_model_dense_1_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
SquareSquaremul_model_dense_1_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:kg
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
/
_user_specified_namemodel/dense_1/BiasAdd:JF

_output_shapes
: 
,
_user_specified_namemodel/dense_1/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Т
є
6__inference_multi_head_attention_layer_call_fn_9236385	
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
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9235527s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:џџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	9236381:'	#
!
_user_specified_name	9236379:'#
!
_user_specified_name	9236377:'#
!
_user_specified_name	9236375:'#
!
_user_specified_name	9236373:'#
!
_user_specified_name	9236371:'#
!
_user_specified_name	9236369:'#
!
_user_specified_name	9236367:YU
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namekey:[W
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namevalue:R N
+
_output_shapes
:џџџџџџџџџ@

_user_specified_namequery
#
§
D__inference_dense_2_layer_call_and_return_conditional_losses_9235099

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@з
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9235090*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ: : 20
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
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Њ
J
.__inference_activation_6_layer_call_fn_9236871

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_9234899`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
	
Щ
#__inference_U_layer_call_fn_9234696
	ou1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCall	ou1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_U_layer_call_and_return_conditional_losses_9234667o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9234692:'#
!
_user_specified_name	9234690:'#
!
_user_specified_name	9234688:'#
!
_user_specified_name	9234686:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	ou1_input
ё	
щ
8__inference_spatial_transformation_layer_call_fn_9234577
spatial_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234551s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9234573:'#
!
_user_specified_name	9234571:'#
!
_user_specified_name	9234569:'#
!
_user_specified_name	9234567:a ]
+
_output_shapes
:џџџџџџџџџ
.
_user_specified_namespatial_layer1_input
М
`
D__inference_flatten_layer_call_and_return_conditional_losses_9236528

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ы

%__inference_op1_layer_call_fn_9236856

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_op1_layer_call_and_return_conditional_losses_9234881o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236852:'#
!
_user_specified_name	9236850:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Њ
J
.__inference_activation_3_layer_call_fn_9236760

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_9234664`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
К
J
.__inference_activation_1_layer_call_fn_9236686

inputs
identityЛ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_9234532d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
е
l
@__inference_add_layer_call_and_return_conditional_losses_9236469
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:џџџџџџџџџ@S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ@:џџџџџџџџџ@:UQ
+
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs_0
ѓ	
ё
@__inference_op2_layer_call_and_return_conditional_losses_9236903

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѕ

*__inference_output_v_layer_call_fn_9236556

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_v_layer_call_and_return_conditional_losses_9235422o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236552:'#
!
_user_specified_name	9236550:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
н
b
F__inference_rescaling_layer_call_and_return_conditional_losses_9236069

inputs
identityH
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџQ
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: N
mulMulinputsCast:y:0*
T0*'
_output_shapes
:џџџџџџџџџS
addAddV2mul:z:0
Cast_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ц

$__inference_internal_grad_fn_9237257
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
к
g
I__inference_activation_2_layer_call_and_return_conditional_losses_9236736

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236727*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѓ	
ё
@__inference_ov2_layer_call_and_return_conditional_losses_9234778

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѓ	
ё
@__inference_ou2_layer_call_and_return_conditional_losses_9236755

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѕ

*__inference_output_p_layer_call_fn_9236575

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_p_layer_call_and_return_conditional_losses_9235407o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236571:'#
!
_user_specified_name	9236569:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
к
g
I__inference_activation_2_layer_call_and_return_conditional_losses_9234635

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234626*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
е

$__inference_internal_grad_fn_9237581
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@b
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:]Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0


K__inference_spatial_layer2_layer_call_and_return_conditional_losses_9234514

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
м
Р
$__inference_internal_grad_fn_9238067
result_grads_0
result_grads_1
result_grads_2!
mul_model_u_activation_3_beta
mul_model_u_ou2_biasadd
identity

identity_1
mulMulmul_model_u_activation_3_betamul_model_u_ou2_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
mul_1Mulmul_model_u_activation_3_betamul_model_u_ou2_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
SquareSquaremul_model_u_ou2_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:\X
'
_output_shapes
:џџџџџџџџџ@
-
_user_specified_namemodel/U/ou2/BiasAdd:QM

_output_shapes
: 
3
_user_specified_namemodel/U/activation_3/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
к
g
I__inference_activation_5_layer_call_and_return_conditional_losses_9234796

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234787*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ы

%__inference_ou1_layer_call_fn_9236708

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ou1_layer_call_and_return_conditional_losses_9234617o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236704:'#
!
_user_specified_name	9236702:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ќ
g
-__inference_concatenate_layer_call_fn_9236091
inputs_0
inputs_1
inputs_2
identityЮ
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_9235059`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Ц

$__inference_internal_grad_fn_9237203
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Ї

)__inference_dense_1_layer_call_fn_9236252

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_9235210|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236248:'#
!
_user_specified_name	9236246:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
#
ћ
B__inference_dense_layer_call_and_return_conditional_losses_9235143

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@з
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9235134*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ: : 20
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
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
э

`
D__inference_reshape_layer_call_and_return_conditional_losses_9236147

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
џџџџџџџџџ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Ј
E
)__inference_reshape_layer_call_fn_9236134

inputs
identityЖ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_9235161d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
е

$__inference_internal_grad_fn_9237527
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@b
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:]Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
м
Р
$__inference_internal_grad_fn_9237959
result_grads_0
result_grads_1
result_grads_2!
mul_model_p_activation_7_beta
mul_model_p_op2_biasadd
identity

identity_1
mulMulmul_model_p_activation_7_betamul_model_p_op2_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
mul_1Mulmul_model_p_activation_7_betamul_model_p_op2_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
SquareSquaremul_model_p_op2_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:\X
'
_output_shapes
:џџџџџџџџџ@
-
_user_specified_namemodel/P/op2/BiasAdd:QM

_output_shapes
: 
3
_user_specified_namemodel/P/activation_7/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
е

$__inference_internal_grad_fn_9237554
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@b
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:]Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
#
§
D__inference_dense_1_layer_call_and_return_conditional_losses_9235210

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@з
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9235201*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ@: : 20
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
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Є
G
+__inference_rescaling_layer_call_fn_9236059

inputs
identityД
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_9235032`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ы

%__inference_ov2_layer_call_fn_9236819

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ov2_layer_call_and_return_conditional_losses_9234778o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236815:'#
!
_user_specified_name	9236813:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ц

$__inference_internal_grad_fn_9237095
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0


>__inference_P_layer_call_and_return_conditional_losses_9234931
	op1_input
op1_9234882:@@
op1_9234884:@
op2_9234911:@@
op2_9234913:@
identityЂop1/StatefulPartitionedCallЂop2/StatefulPartitionedCallх
op1/StatefulPartitionedCallStatefulPartitionedCall	op1_inputop1_9234882op1_9234884*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_op1_layer_call_and_return_conditional_losses_9234881т
activation_6/PartitionedCallPartitionedCall$op1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_9234899
op2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0op2_9234911op2_9234913*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_op2_layer_call_and_return_conditional_losses_9234910т
activation_7/PartitionedCallPartitionedCall$op2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_9234928t
IdentityIdentity%activation_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@^
NoOpNoOp^op1/StatefulPartitionedCall^op2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 2:
op1/StatefulPartitionedCallop1/StatefulPartitionedCall2:
op2/StatefulPartitionedCallop2/StatefulPartitionedCall:'#
!
_user_specified_name	9234913:'#
!
_user_specified_name	9234911:'#
!
_user_specified_name	9234884:'#
!
_user_specified_name	9234882:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	op1_input
М
`
D__inference_flatten_layer_call_and_return_conditional_losses_9235369

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Љ
&
"__inference__wrapped_model_9234433
x_layer
y_layer
t_layer
	xbc_layer
	ybc_layer
	tbc_layer
	ubc_layer
	vbc_layer
	pbc_layerA
/model_dense_2_tensordot_readvariableop_resource:@;
-model_dense_2_biasadd_readvariableop_resource:@?
-model_dense_tensordot_readvariableop_resource:@9
+model_dense_biasadd_readvariableop_resource:@_
Mmodel_spatial_transformation_spatial_layer1_tensordot_readvariableop_resource:@Y
Kmodel_spatial_transformation_spatial_layer1_biasadd_readvariableop_resource:@_
Mmodel_spatial_transformation_spatial_layer2_tensordot_readvariableop_resource:@@Y
Kmodel_spatial_transformation_spatial_layer2_biasadd_readvariableop_resource:@A
/model_dense_1_tensordot_readvariableop_resource:@@;
-model_dense_1_biasadd_readvariableop_resource:@A
/model_dense_3_tensordot_readvariableop_resource:@@;
-model_dense_3_biasadd_readvariableop_resource:@\
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:@@N
<model_multi_head_attention_query_add_readvariableop_resource:@Z
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:@@L
:model_multi_head_attention_key_add_readvariableop_resource:@\
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:@@N
<model_multi_head_attention_value_add_readvariableop_resource:@g
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@U
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:@A
/model_dense_4_tensordot_readvariableop_resource:@@;
-model_dense_4_biasadd_readvariableop_resource:@<
*model_p_op1_matmul_readvariableop_resource:@@9
+model_p_op1_biasadd_readvariableop_resource:@<
*model_p_op2_matmul_readvariableop_resource:@@9
+model_p_op2_biasadd_readvariableop_resource:@<
*model_v_ov1_matmul_readvariableop_resource:@@9
+model_v_ov1_biasadd_readvariableop_resource:@<
*model_v_ov2_matmul_readvariableop_resource:@@9
+model_v_ov2_biasadd_readvariableop_resource:@<
*model_u_ou1_matmul_readvariableop_resource:@@9
+model_u_ou1_biasadd_readvariableop_resource:@<
*model_u_ou2_matmul_readvariableop_resource:@@9
+model_u_ou2_biasadd_readvariableop_resource:@?
-model_output_p_matmul_readvariableop_resource:@<
.model_output_p_biasadd_readvariableop_resource:?
-model_output_v_matmul_readvariableop_resource:@<
.model_output_v_biasadd_readvariableop_resource:?
-model_output_u_matmul_readvariableop_resource:@<
.model_output_u_biasadd_readvariableop_resource:
identity

identity_1

identity_2Ђ"model/P/op1/BiasAdd/ReadVariableOpЂ!model/P/op1/MatMul/ReadVariableOpЂ"model/P/op2/BiasAdd/ReadVariableOpЂ!model/P/op2/MatMul/ReadVariableOpЂ"model/U/ou1/BiasAdd/ReadVariableOpЂ!model/U/ou1/MatMul/ReadVariableOpЂ"model/U/ou2/BiasAdd/ReadVariableOpЂ!model/U/ou2/MatMul/ReadVariableOpЂ"model/V/ov1/BiasAdd/ReadVariableOpЂ!model/V/ov1/MatMul/ReadVariableOpЂ"model/V/ov2/BiasAdd/ReadVariableOpЂ!model/V/ov2/MatMul/ReadVariableOpЂ"model/dense/BiasAdd/ReadVariableOpЂ$model/dense/Tensordot/ReadVariableOpЂ$model/dense_1/BiasAdd/ReadVariableOpЂ&model/dense_1/Tensordot/ReadVariableOpЂ$model/dense_2/BiasAdd/ReadVariableOpЂ&model/dense_2/Tensordot/ReadVariableOpЂ$model/dense_3/BiasAdd/ReadVariableOpЂ&model/dense_3/Tensordot/ReadVariableOpЂ$model/dense_4/BiasAdd/ReadVariableOpЂ&model/dense_4/Tensordot/ReadVariableOpЂ>model/multi_head_attention/attention_output/add/ReadVariableOpЂHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpЂ1model/multi_head_attention/key/add/ReadVariableOpЂ;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpЂ3model/multi_head_attention/query/add/ReadVariableOpЂ=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpЂ3model/multi_head_attention/value/add/ReadVariableOpЂ=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpЂ%model/output_p/BiasAdd/ReadVariableOpЂ$model/output_p/MatMul/ReadVariableOpЂ%model/output_u/BiasAdd/ReadVariableOpЂ$model/output_u/MatMul/ReadVariableOpЂ%model/output_v/BiasAdd/ReadVariableOpЂ$model/output_v/MatMul/ReadVariableOpЂBmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpЂDmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpЂBmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpЂDmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOpZ
model/rescaling_1/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :q
model/rescaling_1/CastCast!model/rescaling_1/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: e
model/rescaling_1/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџu
model/rescaling_1/Cast_1Cast#model/rescaling_1/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: 
model/rescaling_1/mulMul	tbc_layermodel/rescaling_1/Cast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
model/rescaling_1/addAddV2model/rescaling_1/mul:z:0model/rescaling_1/Cast_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџX
model/rescaling/Cast/xConst*
_output_shapes
: *
dtype0*
value	B :m
model/rescaling/CastCastmodel/rescaling/Cast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: c
model/rescaling/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџq
model/rescaling/Cast_1Cast!model/rescaling/Cast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: o
model/rescaling/mulMult_layermodel/rescaling/Cast:y:0*
T0*'
_output_shapes
:џџџџџџџџџ
model/rescaling/addAddV2model/rescaling/mul:z:0model/rescaling/Cast_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџa
model/concatenate_2/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Й
model/concatenate_2/concatConcatV2	ubc_layer	vbc_layer	pbc_layer(model/concatenate_2/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџa
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Щ
model/concatenate_1/concatConcatV2	xbc_layer	ybc_layermodel/rescaling_1/add:z:0(model/concatenate_1/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ_
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :В
model/concatenate/concatConcatV2x_layery_layermodel/rescaling/add:z:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџ
&model/dense_2/Tensordot/ReadVariableOpReadVariableOp/model_dense_2_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0f
model/dense_2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       ~
model/dense_2/Tensordot/ShapeShape#model/concatenate_2/concat:output:0*
T0*
_output_shapes
::эЯg
%model/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
 model/dense_2/Tensordot/GatherV2GatherV2&model/dense_2/Tensordot/Shape:output:0%model/dense_2/Tensordot/free:output:0.model/dense_2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
"model/dense_2/Tensordot/GatherV2_1GatherV2&model/dense_2/Tensordot/Shape:output:0%model/dense_2/Tensordot/axes:output:00model/dense_2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_2/Tensordot/ProdProd)model/dense_2/Tensordot/GatherV2:output:0&model/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_2/Tensordot/Prod_1Prod+model/dense_2/Tensordot/GatherV2_1:output:0(model/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : д
model/dense_2/Tensordot/concatConcatV2%model/dense_2/Tensordot/free:output:0%model/dense_2/Tensordot/axes:output:0,model/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
model/dense_2/Tensordot/stackPack%model/dense_2/Tensordot/Prod:output:0'model/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Л
!model/dense_2/Tensordot/transpose	Transpose#model/concatenate_2/concat:output:0'model/dense_2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџД
model/dense_2/Tensordot/ReshapeReshape%model/dense_2/Tensordot/transpose:y:0&model/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџД
model/dense_2/Tensordot/MatMulMatMul(model/dense_2/Tensordot/Reshape:output:0.model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
model/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
 model/dense_2/Tensordot/concat_1ConcatV2)model/dense_2/Tensordot/GatherV2:output:0(model/dense_2/Tensordot/Const_2:output:0.model/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ж
model/dense_2/TensordotReshape(model/dense_2/Tensordot/MatMul:product:0)model/dense_2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Џ
model/dense_2/BiasAddBiasAdd model/dense_2/Tensordot:output:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@W
model/dense_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense_2/mulMulmodel/dense_2/beta:output:0model/dense_2/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@v
model/dense_2/SigmoidSigmoidmodel/dense_2/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
model/dense_2/mul_1Mulmodel/dense_2/BiasAdd:output:0model/dense_2/Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
model/dense_2/IdentityIdentitymodel/dense_2/mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
model/dense_2/IdentityN	IdentityNmodel/dense_2/mul_1:z:0model/dense_2/BiasAdd:output:0model/dense_2/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234062*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: 
$model/dense/Tensordot/ReadVariableOpReadVariableOp-model_dense_tensordot_readvariableop_resource*
_output_shapes

:@*
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
valueB"       |
model/dense/Tensordot/ShapeShape#model/concatenate_1/concat:output:0*
T0*
_output_shapes
::эЯe
#model/dense/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
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
value	B : я
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
valueB: 
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:З
model/dense/Tensordot/transpose	Transpose#model/concatenate_1/concat:output:0%model/dense/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџЎ
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџЎ
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@g
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:А
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Љ
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@U
model/dense/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense/mulMulmodel/dense/beta:output:0model/dense/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
model/dense/SigmoidSigmoidmodel/dense/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
model/dense/mul_1Mulmodel/dense/BiasAdd:output:0model/dense/Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@v
model/dense/IdentityIdentitymodel/dense/mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
model/dense/IdentityN	IdentityNmodel/dense/mul_1:z:0model/dense/BiasAdd:output:0model/dense/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234097*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: r
model/reshape/ShapeShape!model/concatenate/concat:output:0*
T0*
_output_shapes
::эЯk
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
valueB:
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
џџџџџџџџџЧ
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:
model/reshape/ReshapeReshape!model/concatenate/concat:output:0$model/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџв
Dmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpReadVariableOpMmodel_spatial_transformation_spatial_layer1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0
:model/spatial_transformation/spatial_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
:model/spatial_transformation/spatial_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       
;model/spatial_transformation/spatial_layer1/Tensordot/ShapeShapemodel/reshape/Reshape:output:0*
T0*
_output_shapes
::эЯ
Cmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
>model/spatial_transformation/spatial_layer1/Tensordot/GatherV2GatherV2Dmodel/spatial_transformation/spatial_layer1/Tensordot/Shape:output:0Cmodel/spatial_transformation/spatial_layer1/Tensordot/free:output:0Lmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Emodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
@model/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1GatherV2Dmodel/spatial_transformation/spatial_layer1/Tensordot/Shape:output:0Cmodel/spatial_transformation/spatial_layer1/Tensordot/axes:output:0Nmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
;model/spatial_transformation/spatial_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ђ
:model/spatial_transformation/spatial_layer1/Tensordot/ProdProdGmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2:output:0Dmodel/spatial_transformation/spatial_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: 
=model/spatial_transformation/spatial_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ј
<model/spatial_transformation/spatial_layer1/Tensordot/Prod_1ProdImodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1:output:0Fmodel/spatial_transformation/spatial_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Amodel/spatial_transformation/spatial_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
<model/spatial_transformation/spatial_layer1/Tensordot/concatConcatV2Cmodel/spatial_transformation/spatial_layer1/Tensordot/free:output:0Cmodel/spatial_transformation/spatial_layer1/Tensordot/axes:output:0Jmodel/spatial_transformation/spatial_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:§
;model/spatial_transformation/spatial_layer1/Tensordot/stackPackCmodel/spatial_transformation/spatial_layer1/Tensordot/Prod:output:0Emodel/spatial_transformation/spatial_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:щ
?model/spatial_transformation/spatial_layer1/Tensordot/transpose	Transposemodel/reshape/Reshape:output:0Emodel/spatial_transformation/spatial_layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ
=model/spatial_transformation/spatial_layer1/Tensordot/ReshapeReshapeCmodel/spatial_transformation/spatial_layer1/Tensordot/transpose:y:0Dmodel/spatial_transformation/spatial_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
<model/spatial_transformation/spatial_layer1/Tensordot/MatMulMatMulFmodel/spatial_transformation/spatial_layer1/Tensordot/Reshape:output:0Lmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
=model/spatial_transformation/spatial_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@
Cmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
>model/spatial_transformation/spatial_layer1/Tensordot/concat_1ConcatV2Gmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2:output:0Fmodel/spatial_transformation/spatial_layer1/Tensordot/Const_2:output:0Lmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
5model/spatial_transformation/spatial_layer1/TensordotReshapeFmodel/spatial_transformation/spatial_layer1/Tensordot/MatMul:product:0Gmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ъ
Bmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpReadVariableOpKmodel_spatial_transformation_spatial_layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
3model/spatial_transformation/spatial_layer1/BiasAddBiasAdd>model/spatial_transformation/spatial_layer1/Tensordot:output:0Jmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@q
,model/spatial_transformation/activation/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?н
+model/spatial_transformation/activation/mulMul5model/spatial_transformation/activation/beta:output:0<model/spatial_transformation/spatial_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ё
/model/spatial_transformation/activation/SigmoidSigmoid/model/spatial_transformation/activation/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@н
-model/spatial_transformation/activation/mul_1Mul<model/spatial_transformation/spatial_layer1/BiasAdd:output:03model/spatial_transformation/activation/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ѕ
0model/spatial_transformation/activation/IdentityIdentity1model/spatial_transformation/activation/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@щ
1model/spatial_transformation/activation/IdentityN	IdentityN1model/spatial_transformation/activation/mul_1:z:0<model/spatial_transformation/spatial_layer1/BiasAdd:output:05model/spatial_transformation/activation/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234141*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: в
Dmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOpReadVariableOpMmodel_spatial_transformation_spatial_layer2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0
:model/spatial_transformation/spatial_layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:
:model/spatial_transformation/spatial_layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Г
;model/spatial_transformation/spatial_layer2/Tensordot/ShapeShape:model/spatial_transformation/activation/IdentityN:output:0*
T0*
_output_shapes
::эЯ
Cmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ы
>model/spatial_transformation/spatial_layer2/Tensordot/GatherV2GatherV2Dmodel/spatial_transformation/spatial_layer2/Tensordot/Shape:output:0Cmodel/spatial_transformation/spatial_layer2/Tensordot/free:output:0Lmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
Emodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : я
@model/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1GatherV2Dmodel/spatial_transformation/spatial_layer2/Tensordot/Shape:output:0Cmodel/spatial_transformation/spatial_layer2/Tensordot/axes:output:0Nmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:
;model/spatial_transformation/spatial_layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: ђ
:model/spatial_transformation/spatial_layer2/Tensordot/ProdProdGmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2:output:0Dmodel/spatial_transformation/spatial_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: 
=model/spatial_transformation/spatial_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: ј
<model/spatial_transformation/spatial_layer2/Tensordot/Prod_1ProdImodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1:output:0Fmodel/spatial_transformation/spatial_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: 
Amodel/spatial_transformation/spatial_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ь
<model/spatial_transformation/spatial_layer2/Tensordot/concatConcatV2Cmodel/spatial_transformation/spatial_layer2/Tensordot/free:output:0Cmodel/spatial_transformation/spatial_layer2/Tensordot/axes:output:0Jmodel/spatial_transformation/spatial_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:§
;model/spatial_transformation/spatial_layer2/Tensordot/stackPackCmodel/spatial_transformation/spatial_layer2/Tensordot/Prod:output:0Emodel/spatial_transformation/spatial_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
?model/spatial_transformation/spatial_layer2/Tensordot/transpose	Transpose:model/spatial_transformation/activation/IdentityN:output:0Emodel/spatial_transformation/spatial_layer2/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@
=model/spatial_transformation/spatial_layer2/Tensordot/ReshapeReshapeCmodel/spatial_transformation/spatial_layer2/Tensordot/transpose:y:0Dmodel/spatial_transformation/spatial_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
<model/spatial_transformation/spatial_layer2/Tensordot/MatMulMatMulFmodel/spatial_transformation/spatial_layer2/Tensordot/Reshape:output:0Lmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
=model/spatial_transformation/spatial_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@
Cmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
>model/spatial_transformation/spatial_layer2/Tensordot/concat_1ConcatV2Gmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2:output:0Fmodel/spatial_transformation/spatial_layer2/Tensordot/Const_2:output:0Lmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
5model/spatial_transformation/spatial_layer2/TensordotReshapeFmodel/spatial_transformation/spatial_layer2/Tensordot/MatMul:product:0Gmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ъ
Bmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpReadVariableOpKmodel_spatial_transformation_spatial_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
3model/spatial_transformation/spatial_layer2/BiasAddBiasAdd>model/spatial_transformation/spatial_layer2/Tensordot:output:0Jmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@s
.model/spatial_transformation/activation_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?с
-model/spatial_transformation/activation_1/mulMul7model/spatial_transformation/activation_1/beta:output:0<model/spatial_transformation/spatial_layer2/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Ѕ
1model/spatial_transformation/activation_1/SigmoidSigmoid1model/spatial_transformation/activation_1/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@с
/model/spatial_transformation/activation_1/mul_1Mul<model/spatial_transformation/spatial_layer2/BiasAdd:output:05model/spatial_transformation/activation_1/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Љ
2model/spatial_transformation/activation_1/IdentityIdentity3model/spatial_transformation/activation_1/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@я
3model/spatial_transformation/activation_1/IdentityN	IdentityN3model/spatial_transformation/activation_1/mul_1:z:0<model/spatial_transformation/spatial_layer2/BiasAdd:output:07model/spatial_transformation/activation_1/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234176*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: 
&model/dense_1/Tensordot/ReadVariableOpReadVariableOp/model_dense_1_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0f
model/dense_1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       y
model/dense_1/Tensordot/ShapeShapemodel/dense/IdentityN:output:0*
T0*
_output_shapes
::эЯg
%model/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
 model/dense_1/Tensordot/GatherV2GatherV2&model/dense_1/Tensordot/Shape:output:0%model/dense_1/Tensordot/free:output:0.model/dense_1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
"model/dense_1/Tensordot/GatherV2_1GatherV2&model/dense_1/Tensordot/Shape:output:0%model/dense_1/Tensordot/axes:output:00model/dense_1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_1/Tensordot/ProdProd)model/dense_1/Tensordot/GatherV2:output:0&model/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_1/Tensordot/Prod_1Prod+model/dense_1/Tensordot/GatherV2_1:output:0(model/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : д
model/dense_1/Tensordot/concatConcatV2%model/dense_1/Tensordot/free:output:0%model/dense_1/Tensordot/axes:output:0,model/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
model/dense_1/Tensordot/stackPack%model/dense_1/Tensordot/Prod:output:0'model/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ж
!model/dense_1/Tensordot/transpose	Transposemodel/dense/IdentityN:output:0'model/dense_1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Д
model/dense_1/Tensordot/ReshapeReshape%model/dense_1/Tensordot/transpose:y:0&model/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџД
model/dense_1/Tensordot/MatMulMatMul(model/dense_1/Tensordot/Reshape:output:0.model/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
model/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
 model/dense_1/Tensordot/concat_1ConcatV2)model/dense_1/Tensordot/GatherV2:output:0(model/dense_1/Tensordot/Const_2:output:0.model/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ж
model/dense_1/TensordotReshape(model/dense_1/Tensordot/MatMul:product:0)model/dense_1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Џ
model/dense_1/BiasAddBiasAdd model/dense_1/Tensordot:output:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@W
model/dense_1/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense_1/mulMulmodel/dense_1/beta:output:0model/dense_1/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@v
model/dense_1/SigmoidSigmoidmodel/dense_1/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
model/dense_1/mul_1Mulmodel/dense_1/BiasAdd:output:0model/dense_1/Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
model/dense_1/IdentityIdentitymodel/dense_1/mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
model/dense_1/IdentityN	IdentityNmodel/dense_1/mul_1:z:0model/dense_1/BiasAdd:output:0model/dense_1/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234211*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: 
&model/dense_3/Tensordot/ReadVariableOpReadVariableOp/model_dense_3_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0f
model/dense_3/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_3/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       {
model/dense_3/Tensordot/ShapeShape model/dense_2/IdentityN:output:0*
T0*
_output_shapes
::эЯg
%model/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
 model/dense_3/Tensordot/GatherV2GatherV2&model/dense_3/Tensordot/Shape:output:0%model/dense_3/Tensordot/free:output:0.model/dense_3/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_3/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
"model/dense_3/Tensordot/GatherV2_1GatherV2&model/dense_3/Tensordot/Shape:output:0%model/dense_3/Tensordot/axes:output:00model/dense_3/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_3/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_3/Tensordot/ProdProd)model/dense_3/Tensordot/GatherV2:output:0&model/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_3/Tensordot/Prod_1Prod+model/dense_3/Tensordot/GatherV2_1:output:0(model/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : д
model/dense_3/Tensordot/concatConcatV2%model/dense_3/Tensordot/free:output:0%model/dense_3/Tensordot/axes:output:0,model/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
model/dense_3/Tensordot/stackPack%model/dense_3/Tensordot/Prod:output:0'model/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:И
!model/dense_3/Tensordot/transpose	Transpose model/dense_2/IdentityN:output:0'model/dense_3/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Д
model/dense_3/Tensordot/ReshapeReshape%model/dense_3/Tensordot/transpose:y:0&model/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџД
model/dense_3/Tensordot/MatMulMatMul(model/dense_3/Tensordot/Reshape:output:0.model/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
model/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
 model/dense_3/Tensordot/concat_1ConcatV2)model/dense_3/Tensordot/GatherV2:output:0(model/dense_3/Tensordot/Const_2:output:0.model/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Ж
model/dense_3/TensordotReshape(model/dense_3/Tensordot/MatMul:product:0)model/dense_3/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Џ
model/dense_3/BiasAddBiasAdd model/dense_3/Tensordot:output:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@W
model/dense_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense_3/mulMulmodel/dense_3/beta:output:0model/dense_3/BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@v
model/dense_3/SigmoidSigmoidmodel/dense_3/mul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
model/dense_3/mul_1Mulmodel/dense_3/BiasAdd:output:0model/dense_3/Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@z
model/dense_3/IdentityIdentitymodel/dense_3/mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
model/dense_3/IdentityN	IdentityNmodel/dense_3/mul_1:z:0model/dense_3/BiasAdd:output:0model/dense_3/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234246*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: Ш
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0
.model/multi_head_attention/query/einsum/EinsumEinsum<model/spatial_transformation/activation_1/IdentityN:output:0Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationabc,cde->abdeА
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0с
(model/multi_head_attention/query/add/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@Ф
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0
,model/multi_head_attention/key/einsum/EinsumEinsum model/dense_1/IdentityN:output:0Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdeЌ
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0ф
&model/multi_head_attention/key/add/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@Ш
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0
.model/multi_head_attention/value/einsum/EinsumEinsum model/dense_3/IdentityN:output:0Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdeА
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0ъ
(model/multi_head_attention/value/add/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@e
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >И
model/multi_head_attention/MulMul,model/multi_head_attention/query/add/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@э
(model/multi_head_attention/einsum/EinsumEinsum*model/multi_head_attention/key/add/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
equationaecd,abcd->acbeЋ
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџА
+model/multi_head_attention/dropout/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџњ
*model/multi_head_attention/einsum_1/EinsumEinsum4model/multi_head_attention/dropout/Identity:output:0,model/multi_head_attention/value/add/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationacbe,aecd->abcdо
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0І
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equationabcd,cde->abeТ
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0ў
3model/multi_head_attention/attention_output/add/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@У
model/add/addAddV27model/multi_head_attention/attention_output/add/add:z:0<model/spatial_transformation/activation_1/IdentityN:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@
&model/dense_4/Tensordot/ReadVariableOpReadVariableOp/model_dense_4_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0f
model/dense_4/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:m
model/dense_4/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       l
model/dense_4/Tensordot/ShapeShapemodel/add/add:z:0*
T0*
_output_shapes
::эЯg
%model/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ѓ
 model/dense_4/Tensordot/GatherV2GatherV2&model/dense_4/Tensordot/Shape:output:0%model/dense_4/Tensordot/free:output:0.model/dense_4/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:i
'model/dense_4/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ї
"model/dense_4/Tensordot/GatherV2_1GatherV2&model/dense_4/Tensordot/Shape:output:0%model/dense_4/Tensordot/axes:output:00model/dense_4/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:g
model/dense_4/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: 
model/dense_4/Tensordot/ProdProd)model/dense_4/Tensordot/GatherV2:output:0&model/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
model/dense_4/Tensordot/Prod_1Prod+model/dense_4/Tensordot/GatherV2_1:output:0(model/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : д
model/dense_4/Tensordot/concatConcatV2%model/dense_4/Tensordot/free:output:0%model/dense_4/Tensordot/axes:output:0,model/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Ѓ
model/dense_4/Tensordot/stackPack%model/dense_4/Tensordot/Prod:output:0'model/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
: 
!model/dense_4/Tensordot/transpose	Transposemodel/add/add:z:0'model/dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@Д
model/dense_4/Tensordot/ReshapeReshape%model/dense_4/Tensordot/transpose:y:0&model/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџД
model/dense_4/Tensordot/MatMulMatMul(model/dense_4/Tensordot/Reshape:output:0.model/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@i
model/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : п
 model/dense_4/Tensordot/concat_1ConcatV2)model/dense_4/Tensordot/GatherV2:output:0(model/dense_4/Tensordot/Const_2:output:0.model/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:­
model/dense_4/TensordotReshape(model/dense_4/Tensordot/MatMul:product:0)model/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0І
model/dense_4/BiasAddBiasAdd model/dense_4/Tensordot:output:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@W
model/dense_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/dense_4/mulMulmodel/dense_4/beta:output:0model/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@m
model/dense_4/SigmoidSigmoidmodel/dense_4/mul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@
model/dense_4/mul_1Mulmodel/dense_4/BiasAdd:output:0model/dense_4/Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@q
model/dense_4/IdentityIdentitymodel/dense_4/mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@§
model/dense_4/IdentityN	IdentityNmodel/dense_4/mul_1:z:0model/dense_4/BiasAdd:output:0model/dense_4/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234312*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"џџџџ@   
model/flatten/ReshapeReshape model/dense_4/IdentityN:output:0model/flatten/Const:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@
!model/P/op1/MatMul/ReadVariableOpReadVariableOp*model_p_op1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
model/P/op1/MatMulMatMulmodel/flatten/Reshape:output:0)model/P/op1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"model/P/op1/BiasAdd/ReadVariableOpReadVariableOp+model_p_op1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/P/op1/BiasAddBiasAddmodel/P/op1/MatMul:product:0*model/P/op1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@^
model/P/activation_6/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/P/activation_6/mulMul"model/P/activation_6/beta:output:0model/P/op1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
model/P/activation_6/SigmoidSigmoidmodel/P/activation_6/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/P/activation_6/mul_1Mulmodel/P/op1/BiasAdd:output:0 model/P/activation_6/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
model/P/activation_6/IdentityIdentitymodel/P/activation_6/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/P/activation_6/IdentityN	IdentityNmodel/P/activation_6/mul_1:z:0model/P/op1/BiasAdd:output:0"model/P/activation_6/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234329*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: 
!model/P/op2/MatMul/ReadVariableOpReadVariableOp*model_p_op2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Ђ
model/P/op2/MatMulMatMul'model/P/activation_6/IdentityN:output:0)model/P/op2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"model/P/op2/BiasAdd/ReadVariableOpReadVariableOp+model_p_op2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/P/op2/BiasAddBiasAddmodel/P/op2/MatMul:product:0*model/P/op2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@^
model/P/activation_7/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/P/activation_7/mulMul"model/P/activation_7/beta:output:0model/P/op2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
model/P/activation_7/SigmoidSigmoidmodel/P/activation_7/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/P/activation_7/mul_1Mulmodel/P/op2/BiasAdd:output:0 model/P/activation_7/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
model/P/activation_7/IdentityIdentitymodel/P/activation_7/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/P/activation_7/IdentityN	IdentityNmodel/P/activation_7/mul_1:z:0model/P/op2/BiasAdd:output:0"model/P/activation_7/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234344*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: 
!model/V/ov1/MatMul/ReadVariableOpReadVariableOp*model_v_ov1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
model/V/ov1/MatMulMatMulmodel/flatten/Reshape:output:0)model/V/ov1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"model/V/ov1/BiasAdd/ReadVariableOpReadVariableOp+model_v_ov1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/V/ov1/BiasAddBiasAddmodel/V/ov1/MatMul:product:0*model/V/ov1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@^
model/V/activation_4/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/V/activation_4/mulMul"model/V/activation_4/beta:output:0model/V/ov1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
model/V/activation_4/SigmoidSigmoidmodel/V/activation_4/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/V/activation_4/mul_1Mulmodel/V/ov1/BiasAdd:output:0 model/V/activation_4/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
model/V/activation_4/IdentityIdentitymodel/V/activation_4/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/V/activation_4/IdentityN	IdentityNmodel/V/activation_4/mul_1:z:0model/V/ov1/BiasAdd:output:0"model/V/activation_4/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234359*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: 
!model/V/ov2/MatMul/ReadVariableOpReadVariableOp*model_v_ov2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Ђ
model/V/ov2/MatMulMatMul'model/V/activation_4/IdentityN:output:0)model/V/ov2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"model/V/ov2/BiasAdd/ReadVariableOpReadVariableOp+model_v_ov2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/V/ov2/BiasAddBiasAddmodel/V/ov2/MatMul:product:0*model/V/ov2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@^
model/V/activation_5/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/V/activation_5/mulMul"model/V/activation_5/beta:output:0model/V/ov2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
model/V/activation_5/SigmoidSigmoidmodel/V/activation_5/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/V/activation_5/mul_1Mulmodel/V/ov2/BiasAdd:output:0 model/V/activation_5/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
model/V/activation_5/IdentityIdentitymodel/V/activation_5/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/V/activation_5/IdentityN	IdentityNmodel/V/activation_5/mul_1:z:0model/V/ov2/BiasAdd:output:0"model/V/activation_5/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234374*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: 
!model/U/ou1/MatMul/ReadVariableOpReadVariableOp*model_u_ou1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0
model/U/ou1/MatMulMatMulmodel/flatten/Reshape:output:0)model/U/ou1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"model/U/ou1/BiasAdd/ReadVariableOpReadVariableOp+model_u_ou1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/U/ou1/BiasAddBiasAddmodel/U/ou1/MatMul:product:0*model/U/ou1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@^
model/U/activation_2/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/U/activation_2/mulMul"model/U/activation_2/beta:output:0model/U/ou1/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
model/U/activation_2/SigmoidSigmoidmodel/U/activation_2/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/U/activation_2/mul_1Mulmodel/U/ou1/BiasAdd:output:0 model/U/activation_2/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
model/U/activation_2/IdentityIdentitymodel/U/activation_2/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/U/activation_2/IdentityN	IdentityNmodel/U/activation_2/mul_1:z:0model/U/ou1/BiasAdd:output:0"model/U/activation_2/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234389*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: 
!model/U/ou2/MatMul/ReadVariableOpReadVariableOp*model_u_ou2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Ђ
model/U/ou2/MatMulMatMul'model/U/activation_2/IdentityN:output:0)model/U/ou2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@
"model/U/ou2/BiasAdd/ReadVariableOpReadVariableOp+model_u_ou2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
model/U/ou2/BiasAddBiasAddmodel/U/ou2/MatMul:product:0*model/U/ou2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@^
model/U/activation_3/betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?
model/U/activation_3/mulMul"model/U/activation_3/beta:output:0model/U/ou2/BiasAdd:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@w
model/U/activation_3/SigmoidSigmoidmodel/U/activation_3/mul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/U/activation_3/mul_1Mulmodel/U/ou2/BiasAdd:output:0 model/U/activation_3/Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@{
model/U/activation_3/IdentityIdentitymodel/U/activation_3/mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@
model/U/activation_3/IdentityN	IdentityNmodel/U/activation_3/mul_1:z:0model/U/ou2/BiasAdd:output:0"model/U/activation_3/beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234404*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: 
$model/output_p/MatMul/ReadVariableOpReadVariableOp-model_output_p_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
model/output_p/MatMulMatMul'model/P/activation_7/IdentityN:output:0,model/output_p/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
%model/output_p/BiasAdd/ReadVariableOpReadVariableOp.model_output_p_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
model/output_p/BiasAddBiasAddmodel/output_p/MatMul:product:0-model/output_p/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
$model/output_v/MatMul/ReadVariableOpReadVariableOp-model_output_v_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
model/output_v/MatMulMatMul'model/V/activation_5/IdentityN:output:0,model/output_v/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
%model/output_v/BiasAdd/ReadVariableOpReadVariableOp.model_output_v_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
model/output_v/BiasAddBiasAddmodel/output_v/MatMul:product:0-model/output_v/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
$model/output_u/MatMul/ReadVariableOpReadVariableOp-model_output_u_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ј
model/output_u/MatMulMatMul'model/U/activation_3/IdentityN:output:0,model/output_u/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ
%model/output_u/BiasAdd/ReadVariableOpReadVariableOp.model_output_u_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0Ѓ
model/output_u/BiasAddBiasAddmodel/output_u/MatMul:product:0-model/output_u/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџn
IdentityIdentitymodel/output_p/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџp

Identity_1Identitymodel/output_u/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџp

Identity_2Identitymodel/output_v/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџг
NoOpNoOp#^model/P/op1/BiasAdd/ReadVariableOp"^model/P/op1/MatMul/ReadVariableOp#^model/P/op2/BiasAdd/ReadVariableOp"^model/P/op2/MatMul/ReadVariableOp#^model/U/ou1/BiasAdd/ReadVariableOp"^model/U/ou1/MatMul/ReadVariableOp#^model/U/ou2/BiasAdd/ReadVariableOp"^model/U/ou2/MatMul/ReadVariableOp#^model/V/ov1/BiasAdd/ReadVariableOp"^model/V/ov1/MatMul/ReadVariableOp#^model/V/ov2/BiasAdd/ReadVariableOp"^model/V/ov2/MatMul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp'^model/dense_1/Tensordot/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp'^model/dense_2/Tensordot/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp'^model/dense_3/Tensordot/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp'^model/dense_4/Tensordot/ReadVariableOp?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOp&^model/output_p/BiasAdd/ReadVariableOp%^model/output_p/MatMul/ReadVariableOp&^model/output_u/BiasAdd/ReadVariableOp%^model/output_u/MatMul/ReadVariableOp&^model/output_v/BiasAdd/ReadVariableOp%^model/output_v/MatMul/ReadVariableOpC^model/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpE^model/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpC^model/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpE^model/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2H
"model/P/op1/BiasAdd/ReadVariableOp"model/P/op1/BiasAdd/ReadVariableOp2F
!model/P/op1/MatMul/ReadVariableOp!model/P/op1/MatMul/ReadVariableOp2H
"model/P/op2/BiasAdd/ReadVariableOp"model/P/op2/BiasAdd/ReadVariableOp2F
!model/P/op2/MatMul/ReadVariableOp!model/P/op2/MatMul/ReadVariableOp2H
"model/U/ou1/BiasAdd/ReadVariableOp"model/U/ou1/BiasAdd/ReadVariableOp2F
!model/U/ou1/MatMul/ReadVariableOp!model/U/ou1/MatMul/ReadVariableOp2H
"model/U/ou2/BiasAdd/ReadVariableOp"model/U/ou2/BiasAdd/ReadVariableOp2F
!model/U/ou2/MatMul/ReadVariableOp!model/U/ou2/MatMul/ReadVariableOp2H
"model/V/ov1/BiasAdd/ReadVariableOp"model/V/ov1/BiasAdd/ReadVariableOp2F
!model/V/ov1/MatMul/ReadVariableOp!model/V/ov1/MatMul/ReadVariableOp2H
"model/V/ov2/BiasAdd/ReadVariableOp"model/V/ov2/BiasAdd/ReadVariableOp2F
!model/V/ov2/MatMul/ReadVariableOp!model/V/ov2/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2P
&model/dense_1/Tensordot/ReadVariableOp&model/dense_1/Tensordot/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2P
&model/dense_2/Tensordot/ReadVariableOp&model/dense_2/Tensordot/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2P
&model/dense_3/Tensordot/ReadVariableOp&model/dense_3/Tensordot/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2P
&model/dense_4/Tensordot/ReadVariableOp&model/dense_4/Tensordot/ReadVariableOp2
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp2N
%model/output_p/BiasAdd/ReadVariableOp%model/output_p/BiasAdd/ReadVariableOp2L
$model/output_p/MatMul/ReadVariableOp$model/output_p/MatMul/ReadVariableOp2N
%model/output_u/BiasAdd/ReadVariableOp%model/output_u/BiasAdd/ReadVariableOp2L
$model/output_u/MatMul/ReadVariableOp$model/output_u/MatMul/ReadVariableOp2N
%model/output_v/BiasAdd/ReadVariableOp%model/output_v/BiasAdd/ReadVariableOp2L
$model/output_v/MatMul/ReadVariableOp$model/output_v/MatMul/ReadVariableOp2
Bmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpBmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOp2
Dmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpDmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOp2
Bmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpBmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOp2
Dmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOpDmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp:(0$
"
_user_specified_name
resource:(/$
"
_user_specified_name
resource:(.$
"
_user_specified_name
resource:(-$
"
_user_specified_name
resource:(,$
"
_user_specified_name
resource:(+$
"
_user_specified_name
resource:(*$
"
_user_specified_name
resource:()$
"
_user_specified_name
resource:(($
"
_user_specified_name
resource:('$
"
_user_specified_name
resource:(&$
"
_user_specified_name
resource:(%$
"
_user_specified_name
resource:($$
"
_user_specified_name
resource:(#$
"
_user_specified_name
resource:("$
"
_user_specified_name
resource:(!$
"
_user_specified_name
resource:( $
"
_user_specified_name
resource:($
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
resource:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Pbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Vbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ubc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Tbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ybc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	T_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Y_layer:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	X_layer
ы

%__inference_op2_layer_call_fn_9236893

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_op2_layer_call_and_return_conditional_losses_9234910o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236889:'#
!
_user_specified_name	9236887:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
 
d
H__inference_rescaling_1_layer_call_and_return_conditional_losses_9235021

inputs
identityH
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџQ
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: [
mulMulinputsCast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
addAddV2mul:z:0
Cast_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ў
g
I__inference_activation_1_layer_call_and_return_conditional_losses_9234532

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
mulMulbeta:output:0inputs*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@W
mul_1MulinputsSigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Л
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234523*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: `

Identity_1IdentityIdentityN:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
к
g
I__inference_activation_4_layer_call_and_return_conditional_losses_9236810

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236801*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Е
Q
%__inference_add_layer_call_fn_9236463
inputs_0
inputs_1
identityП
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_9235318d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ@:џџџџџџџџџ@:UQ
+
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:џџџџџџџџџ@
"
_user_specified_name
inputs_0
Ц

$__inference_internal_grad_fn_9237068
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
ѓ	
ё
@__inference_ou2_layer_call_and_return_conditional_losses_9234646

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs


)__inference_dense_4_layer_call_fn_9236478

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_9235358s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236474:'#
!
_user_specified_name	9236472:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
к
g
I__inference_activation_7_layer_call_and_return_conditional_losses_9236921

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236912*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Н.

Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9235527	
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
identityЂ#attention_output/add/ReadVariableOpЂ-attention_output/einsum/Einsum/ReadVariableOpЂkey/add/ReadVariableOpЂ key/einsum/Einsum/ReadVariableOpЂquery/add/ReadVariableOpЂ"query/einsum/Einsum/ReadVariableOpЂvalue/add/ReadVariableOpЂ"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0А
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Г
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Й
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџz
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџЉ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationacbe,aecd->abcdЈ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0е
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0­
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@Д
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:џџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : : : : : : : 2J
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
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namekey:[W
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namevalue:R N
+
_output_shapes
:џџџџџџџџџ@

_user_specified_namequery
Њ
J
.__inference_activation_7_layer_call_fn_9236908

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_9234928`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Б-
ф

'__inference_model_layer_call_fn_9235791
x_layer
y_layer
t_layer
	xbc_layer
	ybc_layer
	tbc_layer
	ubc_layer
	vbc_layer
	pbc_layer
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
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

unknown_25:@@

unknown_26:@

unknown_27:@@

unknown_28:@

unknown_29:@@

unknown_30:@

unknown_31:@@

unknown_32:@

unknown_33:@

unknown_34:

unknown_35:@

unknown_36:

unknown_37:@

unknown_38:
identity

identity_1

identity_2ЂStatefulPartitionedCallъ
StatefulPartitionedCallStatefulPartitionedCallx_layery_layert_layer	xbc_layer	ybc_layer	tbc_layer	ubc_layer	vbc_layer	pbc_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_9235597o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'0#
!
_user_specified_name	9235783:'/#
!
_user_specified_name	9235781:'.#
!
_user_specified_name	9235779:'-#
!
_user_specified_name	9235777:',#
!
_user_specified_name	9235775:'+#
!
_user_specified_name	9235773:'*#
!
_user_specified_name	9235771:')#
!
_user_specified_name	9235769:'(#
!
_user_specified_name	9235767:''#
!
_user_specified_name	9235765:'&#
!
_user_specified_name	9235763:'%#
!
_user_specified_name	9235761:'$#
!
_user_specified_name	9235759:'##
!
_user_specified_name	9235757:'"#
!
_user_specified_name	9235755:'!#
!
_user_specified_name	9235753:' #
!
_user_specified_name	9235751:'#
!
_user_specified_name	9235749:'#
!
_user_specified_name	9235747:'#
!
_user_specified_name	9235745:'#
!
_user_specified_name	9235743:'#
!
_user_specified_name	9235741:'#
!
_user_specified_name	9235739:'#
!
_user_specified_name	9235737:'#
!
_user_specified_name	9235735:'#
!
_user_specified_name	9235733:'#
!
_user_specified_name	9235731:'#
!
_user_specified_name	9235729:'#
!
_user_specified_name	9235727:'#
!
_user_specified_name	9235725:'#
!
_user_specified_name	9235723:'#
!
_user_specified_name	9235721:'#
!
_user_specified_name	9235719:'#
!
_user_specified_name	9235717:'#
!
_user_specified_name	9235715:'#
!
_user_specified_name	9235713:'#
!
_user_specified_name	9235711:'#
!
_user_specified_name	9235709:'
#
!
_user_specified_name	9235707:'	#
!
_user_specified_name	9235705:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Pbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Vbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ubc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Tbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ybc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	T_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Y_layer:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	X_layer
е

$__inference_internal_grad_fn_9237635
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@b
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:]Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
к
g
I__inference_activation_6_layer_call_and_return_conditional_losses_9236884

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236875*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ј	
і
E__inference_output_v_layer_call_and_return_conditional_losses_9236566

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
 
d
H__inference_rescaling_1_layer_call_and_return_conditional_losses_9236084

inputs
identityH
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџQ
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: [
mulMulinputsCast:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ`
addAddV2mul:z:0
Cast_1:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ\
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
к
g
I__inference_activation_6_layer_call_and_return_conditional_losses_9234899

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234890*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
м
Р
$__inference_internal_grad_fn_9237986
result_grads_0
result_grads_1
result_grads_2!
mul_model_v_activation_4_beta
mul_model_v_ov1_biasadd
identity

identity_1
mulMulmul_model_v_activation_4_betamul_model_v_ov1_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
mul_1Mulmul_model_v_activation_4_betamul_model_v_ov1_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
SquareSquaremul_model_v_ov1_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:\X
'
_output_shapes
:џџџџџџџџџ@
-
_user_specified_namemodel/V/ov1/BiasAdd:QM

_output_shapes
: 
3
_user_specified_namemodel/V/activation_4/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Ц
ђ
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234551
spatial_layer1_input(
spatial_layer1_9234538:@$
spatial_layer1_9234540:@(
spatial_layer2_9234544:@@$
spatial_layer2_9234546:@
identityЂ&spatial_layer1/StatefulPartitionedCallЂ&spatial_layer2/StatefulPartitionedCall 
&spatial_layer1/StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputspatial_layer1_9234538spatial_layer1_9234540*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_9234465э
activation/PartitionedCallPartitionedCall/spatial_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_9234483Џ
&spatial_layer2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0spatial_layer2_9234544spatial_layer2_9234546*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_9234514ё
activation_1/PartitionedCallPartitionedCall/spatial_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_9234532x
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@t
NoOpNoOp'^spatial_layer1/StatefulPartitionedCall'^spatial_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 2P
&spatial_layer1/StatefulPartitionedCall&spatial_layer1/StatefulPartitionedCall2P
&spatial_layer2/StatefulPartitionedCall&spatial_layer2/StatefulPartitionedCall:'#
!
_user_specified_name	9234546:'#
!
_user_specified_name	9234544:'#
!
_user_specified_name	9234540:'#
!
_user_specified_name	9234538:a ]
+
_output_shapes
:џџџџџџџџџ
.
_user_specified_namespatial_layer1_input


>__inference_V_layer_call_and_return_conditional_losses_9234815
	ov1_input
ov1_9234802:@@
ov1_9234804:@
ov2_9234808:@@
ov2_9234810:@
identityЂov1/StatefulPartitionedCallЂov2/StatefulPartitionedCallх
ov1/StatefulPartitionedCallStatefulPartitionedCall	ov1_inputov1_9234802ov1_9234804*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ov1_layer_call_and_return_conditional_losses_9234749т
activation_4/PartitionedCallPartitionedCall$ov1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_9234767
ov2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0ov2_9234808ov2_9234810*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ov2_layer_call_and_return_conditional_losses_9234778т
activation_5/PartitionedCallPartitionedCall$ov2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_9234796t
IdentityIdentity%activation_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@^
NoOpNoOp^ov1/StatefulPartitionedCall^ov2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 2:
ov1/StatefulPartitionedCallov1/StatefulPartitionedCall2:
ov2/StatefulPartitionedCallov2/StatefulPartitionedCall:'#
!
_user_specified_name	9234810:'#
!
_user_specified_name	9234808:'#
!
_user_specified_name	9234804:'#
!
_user_specified_name	9234802:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	ov1_input
Ё

J__inference_concatenate_1_layer_call_and_return_conditional_losses_9235050

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:\X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Т
є
6__inference_multi_head_attention_layer_call_fn_9236362	
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
identityЂStatefulPartitionedCallШ
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9235295s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:џџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	9236358:'	#
!
_user_specified_name	9236356:'#
!
_user_specified_name	9236354:'#
!
_user_specified_name	9236352:'#
!
_user_specified_name	9236350:'#
!
_user_specified_name	9236348:'#
!
_user_specified_name	9236346:'#
!
_user_specified_name	9236344:YU
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namekey:[W
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namevalue:R N
+
_output_shapes
:џџџџџџџџџ@

_user_specified_namequery
э

`
D__inference_reshape_layer_call_and_return_conditional_losses_9235161

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::эЯ]
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
valueB:б
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
џџџџџџџџџ
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ\
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ЃЖ
Ю
#__inference__traced_restore_9238375
file_prefix/
assignvariableop_dense_kernel:@+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_2_kernel:@-
assignvariableop_3_dense_2_bias:@3
!assignvariableop_4_dense_1_kernel:@@-
assignvariableop_5_dense_1_bias:@3
!assignvariableop_6_dense_3_kernel:@@-
assignvariableop_7_dense_3_bias:@3
!assignvariableop_8_dense_4_kernel:@@-
assignvariableop_9_dense_4_bias:@5
#assignvariableop_10_output_u_kernel:@/
!assignvariableop_11_output_u_bias:5
#assignvariableop_12_output_v_kernel:@/
!assignvariableop_13_output_v_bias:5
#assignvariableop_14_output_p_kernel:@/
!assignvariableop_15_output_p_bias:;
)assignvariableop_16_spatial_layer1_kernel:@5
'assignvariableop_17_spatial_layer1_bias:@;
)assignvariableop_18_spatial_layer2_kernel:@@5
'assignvariableop_19_spatial_layer2_bias:@K
5assignvariableop_20_multi_head_attention_query_kernel:@@E
3assignvariableop_21_multi_head_attention_query_bias:@I
3assignvariableop_22_multi_head_attention_key_kernel:@@C
1assignvariableop_23_multi_head_attention_key_bias:@K
5assignvariableop_24_multi_head_attention_value_kernel:@@E
3assignvariableop_25_multi_head_attention_value_bias:@V
@assignvariableop_26_multi_head_attention_attention_output_kernel:@@L
>assignvariableop_27_multi_head_attention_attention_output_bias:@0
assignvariableop_28_ou1_kernel:@@*
assignvariableop_29_ou1_bias:@0
assignvariableop_30_ou2_kernel:@@*
assignvariableop_31_ou2_bias:@0
assignvariableop_32_ov1_kernel:@@*
assignvariableop_33_ov1_bias:@0
assignvariableop_34_ov2_kernel:@@*
assignvariableop_35_ov2_bias:@0
assignvariableop_36_op1_kernel:@@*
assignvariableop_37_op1_bias:@0
assignvariableop_38_op2_kernel:@@*
assignvariableop_39_op2_bias:@
identity_41ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_10ЂAssignVariableOp_11ЂAssignVariableOp_12ЂAssignVariableOp_13ЂAssignVariableOp_14ЂAssignVariableOp_15ЂAssignVariableOp_16ЂAssignVariableOp_17ЂAssignVariableOp_18ЂAssignVariableOp_19ЂAssignVariableOp_2ЂAssignVariableOp_20ЂAssignVariableOp_21ЂAssignVariableOp_22ЂAssignVariableOp_23ЂAssignVariableOp_24ЂAssignVariableOp_25ЂAssignVariableOp_26ЂAssignVariableOp_27ЂAssignVariableOp_28ЂAssignVariableOp_29ЂAssignVariableOp_3ЂAssignVariableOp_30ЂAssignVariableOp_31ЂAssignVariableOp_32ЂAssignVariableOp_33ЂAssignVariableOp_34ЂAssignVariableOp_35ЂAssignVariableOp_36ЂAssignVariableOp_37ЂAssignVariableOp_38ЂAssignVariableOp_39ЂAssignVariableOp_4ЂAssignVariableOp_5ЂAssignVariableOp_6ЂAssignVariableOp_7ЂAssignVariableOp_8ЂAssignVariableOp_9з
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*§
valueѓB№)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHТ
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ю
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*К
_output_shapesЇ
Є:::::::::::::::::::::::::::::::::::::::::*7
dtypes-
+2)[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:А
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:Д
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_2_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_2_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_1_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_1_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:И
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:Ж
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_10AssignVariableOp#assignvariableop_10_output_u_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_11AssignVariableOp!assignvariableop_11_output_u_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_12AssignVariableOp#assignvariableop_12_output_v_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_13AssignVariableOp!assignvariableop_13_output_v_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:М
AssignVariableOp_14AssignVariableOp#assignvariableop_14_output_p_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:К
AssignVariableOp_15AssignVariableOp!assignvariableop_15_output_p_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_16AssignVariableOp)assignvariableop_16_spatial_layer1_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_17AssignVariableOp'assignvariableop_17_spatial_layer1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:Т
AssignVariableOp_18AssignVariableOp)assignvariableop_18_spatial_layer2_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:Р
AssignVariableOp_19AssignVariableOp'assignvariableop_19_spatial_layer2_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_20AssignVariableOp5assignvariableop_20_multi_head_attention_query_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_21AssignVariableOp3assignvariableop_21_multi_head_attention_query_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_22AssignVariableOp3assignvariableop_22_multi_head_attention_key_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:Ъ
AssignVariableOp_23AssignVariableOp1assignvariableop_23_multi_head_attention_key_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:Ю
AssignVariableOp_24AssignVariableOp5assignvariableop_24_multi_head_attention_value_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:Ь
AssignVariableOp_25AssignVariableOp3assignvariableop_25_multi_head_attention_value_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:й
AssignVariableOp_26AssignVariableOp@assignvariableop_26_multi_head_attention_attention_output_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:з
AssignVariableOp_27AssignVariableOp>assignvariableop_27_multi_head_attention_attention_output_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_28AssignVariableOpassignvariableop_28_ou1_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_29AssignVariableOpassignvariableop_29_ou1_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_30AssignVariableOpassignvariableop_30_ou2_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_31AssignVariableOpassignvariableop_31_ou2_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_32AssignVariableOpassignvariableop_32_ov1_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_33AssignVariableOpassignvariableop_33_ov1_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_34AssignVariableOpassignvariableop_34_ov2_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_35AssignVariableOpassignvariableop_35_ov2_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_36AssignVariableOpassignvariableop_36_op1_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_37AssignVariableOpassignvariableop_37_op1_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:З
AssignVariableOp_38AssignVariableOpassignvariableop_38_op2_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:Е
AssignVariableOp_39AssignVariableOpassignvariableop_39_op2_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 П
Identity_40Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_41IdentityIdentity_40:output:0^NoOp_1*
T0*
_output_shapes
: 
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_41Identity_41:output:0*(
_construction_contextkEagerRuntime*e
_input_shapesT
R: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:(($
"
_user_specified_name
op2/bias:*'&
$
_user_specified_name
op2/kernel:(&$
"
_user_specified_name
op1/bias:*%&
$
_user_specified_name
op1/kernel:($$
"
_user_specified_name
ov2/bias:*#&
$
_user_specified_name
ov2/kernel:("$
"
_user_specified_name
ov1/bias:*!&
$
_user_specified_name
ov1/kernel:( $
"
_user_specified_name
ou2/bias:*&
$
_user_specified_name
ou2/kernel:($
"
_user_specified_name
ou1/bias:*&
$
_user_specified_name
ou1/kernel:JF
D
_user_specified_name,*multi_head_attention/attention_output/bias:LH
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?;
9
_user_specified_name!multi_head_attention/value/bias:A=
;
_user_specified_name#!multi_head_attention/value/kernel:=9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:3/
-
_user_specified_namespatial_layer2/bias:51
/
_user_specified_namespatial_layer2/kernel:3/
-
_user_specified_namespatial_layer1/bias:51
/
_user_specified_namespatial_layer1/kernel:-)
'
_user_specified_nameoutput_p/bias:/+
)
_user_specified_nameoutput_p/kernel:-)
'
_user_specified_nameoutput_v/bias:/+
)
_user_specified_nameoutput_v/kernel:-)
'
_user_specified_nameoutput_u/bias:/+
)
_user_specified_nameoutput_u/kernel:,
(
&
_user_specified_namedense_4/bias:.	*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
е

$__inference_internal_grad_fn_9237689
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@b
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:]Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
ј	
і
E__inference_output_v_layer_call_and_return_conditional_losses_9235422

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs


0__inference_spatial_layer2_layer_call_fn_9236651

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_9234514s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236647:'#
!
_user_specified_name	9236645:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ц

$__inference_internal_grad_fn_9237122
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Эy
н
B__inference_model_layer_call_and_return_conditional_losses_9235446
x_layer
y_layer
t_layer
	xbc_layer
	ybc_layer
	tbc_layer
	ubc_layer
	vbc_layer
	pbc_layer!
dense_2_9235100:@
dense_2_9235102:@
dense_9235144:@
dense_9235146:@0
spatial_transformation_9235163:@,
spatial_transformation_9235165:@0
spatial_transformation_9235167:@@,
spatial_transformation_9235169:@!
dense_1_9235211:@@
dense_1_9235213:@!
dense_3_9235255:@@
dense_3_9235257:@2
multi_head_attention_9235296:@@.
multi_head_attention_9235298:@2
multi_head_attention_9235300:@@.
multi_head_attention_9235302:@2
multi_head_attention_9235304:@@.
multi_head_attention_9235306:@2
multi_head_attention_9235308:@@*
multi_head_attention_9235310:@!
dense_4_9235359:@@
dense_4_9235361:@
	p_9235371:@@
	p_9235373:@
	p_9235375:@@
	p_9235377:@
	v_9235380:@@
	v_9235382:@
	v_9235384:@@
	v_9235386:@
	u_9235389:@@
	u_9235391:@
	u_9235393:@@
	u_9235395:@"
output_p_9235408:@
output_p_9235410:"
output_v_9235423:@
output_v_9235425:"
output_u_9235438:@
output_u_9235440:
identity

identity_1

identity_2ЂP/StatefulPartitionedCallЂU/StatefulPartitionedCallЂV/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂ,multi_head_attention/StatefulPartitionedCallЂ output_p/StatefulPartitionedCallЂ output_u/StatefulPartitionedCallЂ output_v/StatefulPartitionedCallЂ.spatial_transformation/StatefulPartitionedCallв
rescaling_1/PartitionedCallPartitionedCall	tbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_9235021П
rescaling/PartitionedCallPartitionedCallt_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_9235032ю
concatenate_2/PartitionedCallPartitionedCall	ubc_layer	vbc_layer	pbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9235041
concatenate_1/PartitionedCallPartitionedCall	xbc_layer	ybc_layer$rescaling_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9235050ђ
concatenate/PartitionedCallPartitionedCallx_layery_layer"rescaling/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_9235059
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_9235100dense_2_9235102*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_9235099
dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_9235144dense_9235146*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9235143м
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_9235161
.spatial_transformation/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0spatial_transformation_9235163spatial_transformation_9235165spatial_transformation_9235167spatial_transformation_9235169*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234535
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_9235211dense_1_9235213*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_9235210Ё
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_9235255dense_3_9235257*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_9235254ё
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:0(dense_3/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0multi_head_attention_9235296multi_head_attention_9235298multi_head_attention_9235300multi_head_attention_9235302multi_head_attention_9235304multi_head_attention_9235306multi_head_attention_9235308multi_head_attention_9235310*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9235295
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:07spatial_transformation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_9235318
dense_4/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0dense_4_9235359dense_4_9235361*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_9235358м
flatten/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9235369
P/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	p_9235371	p_9235373	p_9235375	p_9235377*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_P_layer_call_and_return_conditional_losses_9234931
V/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	v_9235380	v_9235382	v_9235384	v_9235386*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_V_layer_call_and_return_conditional_losses_9234799
U/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	u_9235389	u_9235391	u_9235393	u_9235395*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_U_layer_call_and_return_conditional_losses_9234667
 output_p/StatefulPartitionedCallStatefulPartitionedCall"P/StatefulPartitionedCall:output:0output_p_9235408output_p_9235410*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_p_layer_call_and_return_conditional_losses_9235407
 output_v/StatefulPartitionedCallStatefulPartitionedCall"V/StatefulPartitionedCall:output:0output_v_9235423output_v_9235425*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_v_layer_call_and_return_conditional_losses_9235422
 output_u/StatefulPartitionedCallStatefulPartitionedCall"U/StatefulPartitionedCall:output:0output_u_9235438output_u_9235440*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_u_layer_call_and_return_conditional_losses_9235437x
IdentityIdentity)output_u/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџz

Identity_1Identity)output_v/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџz

Identity_2Identity)output_p/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџч
NoOpNoOp^P/StatefulPartitionedCall^U/StatefulPartitionedCall^V/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall!^output_p/StatefulPartitionedCall!^output_u/StatefulPartitionedCall!^output_v/StatefulPartitionedCall/^spatial_transformation/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 26
P/StatefulPartitionedCallP/StatefulPartitionedCall26
U/StatefulPartitionedCallU/StatefulPartitionedCall26
V/StatefulPartitionedCallV/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2D
 output_p/StatefulPartitionedCall output_p/StatefulPartitionedCall2D
 output_u/StatefulPartitionedCall output_u/StatefulPartitionedCall2D
 output_v/StatefulPartitionedCall output_v/StatefulPartitionedCall2`
.spatial_transformation/StatefulPartitionedCall.spatial_transformation/StatefulPartitionedCall:'0#
!
_user_specified_name	9235440:'/#
!
_user_specified_name	9235438:'.#
!
_user_specified_name	9235425:'-#
!
_user_specified_name	9235423:',#
!
_user_specified_name	9235410:'+#
!
_user_specified_name	9235408:'*#
!
_user_specified_name	9235395:')#
!
_user_specified_name	9235393:'(#
!
_user_specified_name	9235391:''#
!
_user_specified_name	9235389:'&#
!
_user_specified_name	9235386:'%#
!
_user_specified_name	9235384:'$#
!
_user_specified_name	9235382:'##
!
_user_specified_name	9235380:'"#
!
_user_specified_name	9235377:'!#
!
_user_specified_name	9235375:' #
!
_user_specified_name	9235373:'#
!
_user_specified_name	9235371:'#
!
_user_specified_name	9235361:'#
!
_user_specified_name	9235359:'#
!
_user_specified_name	9235310:'#
!
_user_specified_name	9235308:'#
!
_user_specified_name	9235306:'#
!
_user_specified_name	9235304:'#
!
_user_specified_name	9235302:'#
!
_user_specified_name	9235300:'#
!
_user_specified_name	9235298:'#
!
_user_specified_name	9235296:'#
!
_user_specified_name	9235257:'#
!
_user_specified_name	9235255:'#
!
_user_specified_name	9235213:'#
!
_user_specified_name	9235211:'#
!
_user_specified_name	9235169:'#
!
_user_specified_name	9235167:'#
!
_user_specified_name	9235165:'#
!
_user_specified_name	9235163:'#
!
_user_specified_name	9235146:'#
!
_user_specified_name	9235144:'
#
!
_user_specified_name	9235102:'	#
!
_user_specified_name	9235100:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Pbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Vbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ubc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Tbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ybc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	T_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Y_layer:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	X_layer
ѓ	
ё
@__inference_ov1_layer_call_and_return_conditional_losses_9234749

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѓ	
ё
@__inference_ou1_layer_call_and_return_conditional_losses_9236718

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
 

$__inference_internal_grad_fn_9237473
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1h
mulMulmul_betamul_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Y
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@S
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:TP
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0


K__inference_spatial_layer1_layer_call_and_return_conditional_losses_9234465

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
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
:џџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 20
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
:џџџџџџџџџ
 
_user_specified_nameinputs
ё	
щ
8__inference_spatial_transformation_layer_call_fn_9234564
spatial_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234535s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:џџџџџџџџџ: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9234560:'#
!
_user_specified_name	9234558:'#
!
_user_specified_name	9234556:'#
!
_user_specified_name	9234554:a ]
+
_output_shapes
:џџџџџџџџџ
.
_user_specified_namespatial_layer1_input
	
Щ
#__inference_V_layer_call_fn_9234828
	ov1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCall	ov1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_V_layer_call_and_return_conditional_losses_9234799o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9234824:'#
!
_user_specified_name	9234822:'#
!
_user_specified_name	9234820:'#
!
_user_specified_name	9234818:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	ov1_input
Ё

J__inference_concatenate_2_layer_call_and_return_conditional_losses_9235041

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:\X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:\X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ

J__inference_concatenate_2_layer_call_and_return_conditional_losses_9236129
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0


$__inference_internal_grad_fn_9237446
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1g
mulMulmul_beta
mul_inputs^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_1Mulmul_beta
mul_inputs*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@R
SquareSquare
mul_inputs*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:SO
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Ж

H__inference_concatenate_layer_call_and_return_conditional_losses_9235059

inputs
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputsinputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:OK
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
д
Л
$__inference_internal_grad_fn_9237878
result_grads_0
result_grads_1
result_grads_2
mul_model_dense_3_beta
mul_model_dense_3_biasadd
identity

identity_1
mulMulmul_model_dense_3_betamul_model_dense_3_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@~
mul_1Mulmul_model_dense_3_betamul_model_dense_3_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
SquareSquaremul_model_dense_3_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:kg
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
/
_user_specified_namemodel/dense_3/BiasAdd:JF

_output_shapes
: 
,
_user_specified_namemodel/dense_3/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
	
Щ
#__inference_P_layer_call_fn_9234960
	op1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityЂStatefulPartitionedCallѓ
StatefulPartitionedCallStatefulPartitionedCall	op1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_P_layer_call_and_return_conditional_losses_9234931o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9234956:'#
!
_user_specified_name	9234954:'#
!
_user_specified_name	9234952:'#
!
_user_specified_name	9234950:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	op1_input
Ц

$__inference_internal_grad_fn_9237311
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
к
g
I__inference_activation_3_layer_call_and_return_conditional_losses_9234664

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234655*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
к
g
I__inference_activation_5_layer_call_and_return_conditional_losses_9236847

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236838*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѕ

*__inference_output_u_layer_call_fn_9236537

inputs
unknown:@
	unknown_0:
identityЂStatefulPartitionedCallн
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_u_layer_call_and_return_conditional_losses_9235437o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236533:'#
!
_user_specified_name	9236531:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ѓ	
ё
@__inference_ou1_layer_call_and_return_conditional_losses_9234617

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
-
т

%__inference_signature_wrapper_9236054
	pbc_layer
t_layer
	tbc_layer
	ubc_layer
	vbc_layer
x_layer
	xbc_layer
y_layer
	ybc_layer
unknown:@
	unknown_0:@
	unknown_1:@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
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

unknown_25:@@

unknown_26:@

unknown_27:@@

unknown_28:@

unknown_29:@@

unknown_30:@

unknown_31:@@

unknown_32:@

unknown_33:@

unknown_34:

unknown_35:@

unknown_36:

unknown_37:@

unknown_38:
identity

identity_1

identity_2ЂStatefulPartitionedCallЪ
StatefulPartitionedCallStatefulPartitionedCallx_layery_layert_layer	xbc_layer	ybc_layer	tbc_layer	ubc_layer	vbc_layer	pbc_layerunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
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
unknown_25
unknown_26
unknown_27
unknown_28
unknown_29
unknown_30
unknown_31
unknown_32
unknown_33
unknown_34
unknown_35
unknown_36
unknown_37
unknown_38*<
Tin5
321*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ*J
_read_only_resource_inputs,
*(	
 !"#$%&'()*+,-./0*0
config_proto 

CPU

GPU2*0J 8 *+
f&R$
"__inference__wrapped_model_9234433o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_1Identity StatefulPartitionedCall:output:1^NoOp*
T0*'
_output_shapes
:џџџџџџџџџq

Identity_2Identity StatefulPartitionedCall:output:2^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'0#
!
_user_specified_name	9236046:'/#
!
_user_specified_name	9236044:'.#
!
_user_specified_name	9236042:'-#
!
_user_specified_name	9236040:',#
!
_user_specified_name	9236038:'+#
!
_user_specified_name	9236036:'*#
!
_user_specified_name	9236034:')#
!
_user_specified_name	9236032:'(#
!
_user_specified_name	9236030:''#
!
_user_specified_name	9236028:'&#
!
_user_specified_name	9236026:'%#
!
_user_specified_name	9236024:'$#
!
_user_specified_name	9236022:'##
!
_user_specified_name	9236020:'"#
!
_user_specified_name	9236018:'!#
!
_user_specified_name	9236016:' #
!
_user_specified_name	9236014:'#
!
_user_specified_name	9236012:'#
!
_user_specified_name	9236010:'#
!
_user_specified_name	9236008:'#
!
_user_specified_name	9236006:'#
!
_user_specified_name	9236004:'#
!
_user_specified_name	9236002:'#
!
_user_specified_name	9236000:'#
!
_user_specified_name	9235998:'#
!
_user_specified_name	9235996:'#
!
_user_specified_name	9235994:'#
!
_user_specified_name	9235992:'#
!
_user_specified_name	9235990:'#
!
_user_specified_name	9235988:'#
!
_user_specified_name	9235986:'#
!
_user_specified_name	9235984:'#
!
_user_specified_name	9235982:'#
!
_user_specified_name	9235980:'#
!
_user_specified_name	9235978:'#
!
_user_specified_name	9235976:'#
!
_user_specified_name	9235974:'#
!
_user_specified_name	9235972:'
#
!
_user_specified_name	9235970:'	#
!
_user_specified_name	9235968:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ybc_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Y_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	X_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Vbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ubc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Tbc_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	T_layer:_ [
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Pbc_layer
 

$__inference_internal_grad_fn_9237500
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1h
mulMulmul_betamul_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Y
mul_1Mulmul_betamul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@S
SquareSquaremul_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:TP
+
_output_shapes
:џџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
к
g
I__inference_activation_3_layer_call_and_return_conditional_losses_9236773

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236764*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Э
j
@__inference_add_layer_call_and_return_conditional_losses_9235318

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:џџџџџџџџџ@S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:џџџџџџџџџ@:џџџџџџџџџ@:SO
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
м
I
-__inference_rescaling_1_layer_call_fn_9236074

inputs
identityУ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_9235021m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :џџџџџџџџџџџџџџџџџџ:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ј
E
)__inference_flatten_layer_call_fn_9236522

inputs
identityВ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9235369`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


$__inference_internal_grad_fn_9237419
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1g
mulMulmul_beta
mul_inputs^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_1Mulmul_beta
mul_inputs*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@R
SquareSquare
mul_inputs*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:SO
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
ѓ	
ё
@__inference_op1_layer_call_and_return_conditional_losses_9234881

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
ј	
і
E__inference_output_u_layer_call_and_return_conditional_losses_9236547

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
С

H__inference_concatenate_layer_call_and_return_conditional_losses_9236099
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*'
_output_shapes
:џџџџџџџџџW
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*L
_input_shapes;
9:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_2:QM
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:џџџџџџџџџ
"
_user_specified_name
inputs_0
Ѓ

'__inference_dense_layer_call_fn_9236156

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9235143|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236152:'#
!
_user_specified_name	9236150:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
Ћ

J__inference_concatenate_1_layer_call_and_return_conditional_losses_9236114
inputs_0
inputs_1
inputs_2
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :
concatConcatV2inputs_0inputs_1inputs_2concat/axis:output:0*
N*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџd
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0
м
Р
$__inference_internal_grad_fn_9238040
result_grads_0
result_grads_1
result_grads_2!
mul_model_u_activation_2_beta
mul_model_u_ou1_biasadd
identity

identity_1
mulMulmul_model_u_activation_2_betamul_model_u_ou1_biasadd^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@v
mul_1Mulmul_model_u_activation_2_betamul_model_u_ou1_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
SquareSquaremul_model_u_ou1_biasadd*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:\X
'
_output_shapes
:џџџџџџџџџ@
-
_user_specified_namemodel/U/ou1/BiasAdd:QM

_output_shapes
: 
3
_user_specified_namemodel/U/activation_2/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
н
b
F__inference_rescaling_layer_call_and_return_conditional_losses_9235032

inputs
identityH
Cast/xConst*
_output_shapes
: *
dtype0*
value	B :M
CastCastCast/x:output:0*

DstT0*

SrcT0*
_output_shapes
: S
Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB :
џџџџџџџџџQ
Cast_1CastCast_1/x:output:0*

DstT0*

SrcT0*
_output_shapes
: N
mulMulinputsCast:y:0*
T0*'
_output_shapes
:џџџџџџџџџS
addAddV2mul:z:0
Cast_1:y:0*
T0*'
_output_shapes
:џџџџџџџџџO
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:џџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ:O K
'
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ѓ	
ё
@__inference_op2_layer_call_and_return_conditional_losses_9234910

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ц

$__inference_internal_grad_fn_9237041
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0


>__inference_V_layer_call_and_return_conditional_losses_9234799
	ov1_input
ov1_9234750:@@
ov1_9234752:@
ov2_9234779:@@
ov2_9234781:@
identityЂov1/StatefulPartitionedCallЂov2/StatefulPartitionedCallх
ov1/StatefulPartitionedCallStatefulPartitionedCall	ov1_inputov1_9234750ov1_9234752*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ov1_layer_call_and_return_conditional_losses_9234749т
activation_4/PartitionedCallPartitionedCall$ov1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_9234767
ov2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0ov2_9234779ov2_9234781*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ov2_layer_call_and_return_conditional_losses_9234778т
activation_5/PartitionedCallPartitionedCall$ov2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_9234796t
IdentityIdentity%activation_5/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@^
NoOpNoOp^ov1/StatefulPartitionedCall^ov2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 2:
ov1/StatefulPartitionedCallov1/StatefulPartitionedCall2:
ov2/StatefulPartitionedCallov2/StatefulPartitionedCall:'#
!
_user_specified_name	9234781:'#
!
_user_specified_name	9234779:'#
!
_user_specified_name	9234752:'#
!
_user_specified_name	9234750:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	ov1_input
Ц

$__inference_internal_grad_fn_9237176
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
к
g
I__inference_activation_4_layer_call_and_return_conditional_losses_9234767

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?S
mulMulbeta:output:0inputs*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@S
mul_1MulinputsSigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Г
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9234758*<
_output_shapes*
(:џџџџџџџџџ@:џџџџџџџџџ@: \

Identity_1IdentityIdentityN:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs


0__inference_spatial_layer1_layer_call_fn_9236594

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_9234465s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236590:'#
!
_user_specified_name	9236588:S O
+
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Н.

Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9236421	
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
identityЂ#attention_output/add/ReadVariableOpЂ-attention_output/einsum/Einsum/ReadVariableOpЂkey/add/ReadVariableOpЂ key/einsum/Einsum/ReadVariableOpЂquery/add/ReadVariableOpЂ"query/einsum/Einsum/ReadVariableOpЂvalue/add/ReadVariableOpЂ"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0А
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Г
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Й
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџz
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџЉ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationacbe,aecd->abcdЈ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0е
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0­
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@Д
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:џџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : : : : : : : 2J
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
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namekey:[W
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namevalue:R N
+
_output_shapes
:џџџџџџџџџ@

_user_specified_namequery


$__inference_internal_grad_fn_9237392
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1g
mulMulmul_beta
mul_inputs^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_1Mulmul_beta
mul_inputs*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@R
SquareSquare
mul_inputs*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:SO
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
ы

%__inference_ov1_layer_call_fn_9236782

inputs
unknown:@@
	unknown_0:@
identityЂStatefulPartitionedCallи
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ov1_layer_call_and_return_conditional_losses_9234749o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236778:'#
!
_user_specified_name	9236776:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
ў
g
I__inference_activation_1_layer_call_and_return_conditional_losses_9236699

inputs

identity_1I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?W
mulMulbeta:output:0inputs*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@W
mul_1MulinputsSigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Л
	IdentityN	IdentityN	mul_1:z:0inputsbeta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236690*D
_output_shapes2
0:џџџџџџџџџ@:џџџџџџџџџ@: `

Identity_1IdentityIdentityN:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Н.

Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9235295	
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
identityЂ#attention_output/add/ReadVariableOpЂ-attention_output/einsum/Einsum/ReadVariableOpЂkey/add/ReadVariableOpЂ key/einsum/Einsum/ReadVariableOpЂquery/add/ReadVariableOpЂ"query/einsum/Einsum/ReadVariableOpЂvalue/add/ReadVariableOpЂ"value/einsum/Einsum/ReadVariableOp
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0А
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:џџџџџџџџџ@
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Г
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Й
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ@J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:џџџџџџџџџ@
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџ*
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџz
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"џџџџџџџџџџџџџџџџџџЉ
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:џџџџџџџџџ@*
equationacbe,aecd->abcdЈ
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0е
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:џџџџџџџџџ@*
equationabcd,cde->abe
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0­
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:џџџџџџџџџ@o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:џџџџџџџџџ@Д
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:џџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : : : : : : : 2J
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
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namekey:[W
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@

_user_specified_namevalue:R N
+
_output_shapes
:џџџџџџџџџ@

_user_specified_namequery
Ж
H
,__inference_activation_layer_call_fn_9236629

inputs
identityЙ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_9234483d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@:S O
+
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs
Эy
н
B__inference_model_layer_call_and_return_conditional_losses_9235597
x_layer
y_layer
t_layer
	xbc_layer
	ybc_layer
	tbc_layer
	ubc_layer
	vbc_layer
	pbc_layer!
dense_2_9235462:@
dense_2_9235464:@
dense_9235467:@
dense_9235469:@0
spatial_transformation_9235473:@,
spatial_transformation_9235475:@0
spatial_transformation_9235477:@@,
spatial_transformation_9235479:@!
dense_1_9235482:@@
dense_1_9235484:@!
dense_3_9235487:@@
dense_3_9235489:@2
multi_head_attention_9235528:@@.
multi_head_attention_9235530:@2
multi_head_attention_9235532:@@.
multi_head_attention_9235534:@2
multi_head_attention_9235536:@@.
multi_head_attention_9235538:@2
multi_head_attention_9235540:@@*
multi_head_attention_9235542:@!
dense_4_9235546:@@
dense_4_9235548:@
	p_9235552:@@
	p_9235554:@
	p_9235556:@@
	p_9235558:@
	v_9235561:@@
	v_9235563:@
	v_9235565:@@
	v_9235567:@
	u_9235570:@@
	u_9235572:@
	u_9235574:@@
	u_9235576:@"
output_p_9235579:@
output_p_9235581:"
output_v_9235584:@
output_v_9235586:"
output_u_9235589:@
output_u_9235591:
identity

identity_1

identity_2ЂP/StatefulPartitionedCallЂU/StatefulPartitionedCallЂV/StatefulPartitionedCallЂdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCallЂdense_2/StatefulPartitionedCallЂdense_3/StatefulPartitionedCallЂdense_4/StatefulPartitionedCallЂ,multi_head_attention/StatefulPartitionedCallЂ output_p/StatefulPartitionedCallЂ output_u/StatefulPartitionedCallЂ output_v/StatefulPartitionedCallЂ.spatial_transformation/StatefulPartitionedCallв
rescaling_1/PartitionedCallPartitionedCall	tbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_9235021П
rescaling/PartitionedCallPartitionedCallt_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_9235032ю
concatenate_2/PartitionedCallPartitionedCall	ubc_layer	vbc_layer	pbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9235041
concatenate_1/PartitionedCallPartitionedCall	xbc_layer	ybc_layer$rescaling_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9235050ђ
concatenate/PartitionedCallPartitionedCallx_layery_layer"rescaling/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_9235059
dense_2/StatefulPartitionedCallStatefulPartitionedCall&concatenate_2/PartitionedCall:output:0dense_2_9235462dense_2_9235464*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_9235099
dense/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0dense_9235467dense_9235469*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_9235143м
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_9235161
.spatial_transformation/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0spatial_transformation_9235473spatial_transformation_9235475spatial_transformation_9235477spatial_transformation_9235479*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *\
fWRU
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234551
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_9235482dense_1_9235484*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_9235210Ё
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_9235487dense_3_9235489*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_9235254ё
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:0(dense_3/StatefulPartitionedCall:output:0(dense_1/StatefulPartitionedCall:output:0multi_head_attention_9235528multi_head_attention_9235530multi_head_attention_9235532multi_head_attention_9235534multi_head_attention_9235536multi_head_attention_9235538multi_head_attention_9235540multi_head_attention_9235542*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8 *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9235527
add/PartitionedCallPartitionedCall5multi_head_attention/StatefulPartitionedCall:output:07spatial_transformation/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_9235318
dense_4/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0dense_4_9235546dense_4_9235548*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_9235358м
flatten/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_9235369
P/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	p_9235552	p_9235554	p_9235556	p_9235558*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_P_layer_call_and_return_conditional_losses_9234947
V/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	v_9235561	v_9235563	v_9235565	v_9235567*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_V_layer_call_and_return_conditional_losses_9234815
U/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	u_9235570	u_9235572	u_9235574	u_9235576*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *G
fBR@
>__inference_U_layer_call_and_return_conditional_losses_9234683
 output_p/StatefulPartitionedCallStatefulPartitionedCall"P/StatefulPartitionedCall:output:0output_p_9235579output_p_9235581*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_p_layer_call_and_return_conditional_losses_9235407
 output_v/StatefulPartitionedCallStatefulPartitionedCall"V/StatefulPartitionedCall:output:0output_v_9235584output_v_9235586*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_v_layer_call_and_return_conditional_losses_9235422
 output_u/StatefulPartitionedCallStatefulPartitionedCall"U/StatefulPartitionedCall:output:0output_u_9235589output_u_9235591*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *N
fIRG
E__inference_output_u_layer_call_and_return_conditional_losses_9235437x
IdentityIdentity)output_u/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџz

Identity_1Identity)output_v/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџz

Identity_2Identity)output_p/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџч
NoOpNoOp^P/StatefulPartitionedCall^U/StatefulPartitionedCall^V/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall!^output_p/StatefulPartitionedCall!^output_u/StatefulPartitionedCall!^output_v/StatefulPartitionedCall/^spatial_transformation/StatefulPartitionedCall*
_output_shapes
 "!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*о
_input_shapesЬ
Щ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 26
P/StatefulPartitionedCallP/StatefulPartitionedCall26
U/StatefulPartitionedCallU/StatefulPartitionedCall26
V/StatefulPartitionedCallV/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2D
 output_p/StatefulPartitionedCall output_p/StatefulPartitionedCall2D
 output_u/StatefulPartitionedCall output_u/StatefulPartitionedCall2D
 output_v/StatefulPartitionedCall output_v/StatefulPartitionedCall2`
.spatial_transformation/StatefulPartitionedCall.spatial_transformation/StatefulPartitionedCall:'0#
!
_user_specified_name	9235591:'/#
!
_user_specified_name	9235589:'.#
!
_user_specified_name	9235586:'-#
!
_user_specified_name	9235584:',#
!
_user_specified_name	9235581:'+#
!
_user_specified_name	9235579:'*#
!
_user_specified_name	9235576:')#
!
_user_specified_name	9235574:'(#
!
_user_specified_name	9235572:''#
!
_user_specified_name	9235570:'&#
!
_user_specified_name	9235567:'%#
!
_user_specified_name	9235565:'$#
!
_user_specified_name	9235563:'##
!
_user_specified_name	9235561:'"#
!
_user_specified_name	9235558:'!#
!
_user_specified_name	9235556:' #
!
_user_specified_name	9235554:'#
!
_user_specified_name	9235552:'#
!
_user_specified_name	9235548:'#
!
_user_specified_name	9235546:'#
!
_user_specified_name	9235542:'#
!
_user_specified_name	9235540:'#
!
_user_specified_name	9235538:'#
!
_user_specified_name	9235536:'#
!
_user_specified_name	9235534:'#
!
_user_specified_name	9235532:'#
!
_user_specified_name	9235530:'#
!
_user_specified_name	9235528:'#
!
_user_specified_name	9235489:'#
!
_user_specified_name	9235487:'#
!
_user_specified_name	9235484:'#
!
_user_specified_name	9235482:'#
!
_user_specified_name	9235479:'#
!
_user_specified_name	9235477:'#
!
_user_specified_name	9235475:'#
!
_user_specified_name	9235473:'#
!
_user_specified_name	9235469:'#
!
_user_specified_name	9235467:'
#
!
_user_specified_name	9235464:'	#
!
_user_specified_name	9235462:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Pbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Vbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ubc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Tbc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Ybc_layer:_[
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	T_layer:PL
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	Y_layer:P L
'
_output_shapes
:џџџџџџџџџ
!
_user_specified_name	X_layer
#
§
D__inference_dense_1_layer_call_and_return_conditional_losses_9236291

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@з
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236282*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ@: : 20
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
 :џџџџџџџџџџџџџџџџџџ@
 
_user_specified_nameinputs
Ц

$__inference_internal_grad_fn_9237149
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0


>__inference_U_layer_call_and_return_conditional_losses_9234683
	ou1_input
ou1_9234670:@@
ou1_9234672:@
ou2_9234676:@@
ou2_9234678:@
identityЂou1/StatefulPartitionedCallЂou2/StatefulPartitionedCallх
ou1/StatefulPartitionedCallStatefulPartitionedCall	ou1_inputou1_9234670ou1_9234672*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ou1_layer_call_and_return_conditional_losses_9234617т
activation_2/PartitionedCallPartitionedCall$ou1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_9234635
ou2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0ou2_9234676ou2_9234678*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_ou2_layer_call_and_return_conditional_losses_9234646т
activation_3/PartitionedCallPartitionedCall$ou2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_9234664t
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@^
NoOpNoOp^ou1/StatefulPartitionedCall^ou2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:џџџџџџџџџ@: : : : 2:
ou1/StatefulPartitionedCallou1/StatefulPartitionedCall2:
ou2/StatefulPartitionedCallou2/StatefulPartitionedCall:'#
!
_user_specified_name	9234678:'#
!
_user_specified_name	9234676:'#
!
_user_specified_name	9234672:'#
!
_user_specified_name	9234670:R N
'
_output_shapes
:џџџџџџџџџ@
#
_user_specified_name	ou1_input
ѓ	
ё
@__inference_op1_layer_call_and_return_conditional_losses_9236866

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ц

$__inference_internal_grad_fn_9237284
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
ј	
і
E__inference_output_u_layer_call_and_return_conditional_losses_9235437

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
е

$__inference_internal_grad_fn_9237716
result_grads_0
result_grads_1
result_grads_2
mul_beta
mul_biasadd
identity

identity_1q
mulMulmul_betamul_biasadd^result_grads_0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@b
mul_1Mulmul_betamul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
subSubsub/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@_
mul_2Mul	mul_1:z:0sub:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
addAddV2add/Const:output:0	mul_2:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_3MulSigmoid:y:0add:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@\
SquareSquaremul_biasadd*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@g
mul_4Mulresult_grads_0
Square:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@c
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?n
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@a
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: f
mul_7Mulresult_grads_0	mul_3:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_7:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*w
_input_shapesf
d:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: : :џџџџџџџџџџџџџџџџџџ@:]Y
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
!
_user_specified_name	BiasAdd:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:d`
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@
(
_user_specified_nameresult_grads_0
Њ
J
.__inference_activation_4_layer_call_fn_9236797

inputs
identityЗ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:џџџџџџџџџ@* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_9234767`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:џџџџџџџџџ@"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:џџџџџџџџџ@:O K
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs

ѓ
$__inference_internal_grad_fn_9237797
result_grads_0
result_grads_1
result_grads_24
0mul_model_spatial_transformation_activation_beta;
7mul_model_spatial_transformation_spatial_layer1_biasadd
identity

identity_1М
mulMul0mul_model_spatial_transformation_activation_beta7mul_model_spatial_transformation_spatial_layer1_biasadd^result_grads_0*
T0*+
_output_shapes
:џџџџџџџџџ@Q
SigmoidSigmoidmul:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@­
mul_1Mul0mul_model_spatial_transformation_activation_beta7mul_model_spatial_transformation_spatial_layer1_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
subSubsub/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@V
mul_2Mul	mul_1:z:0sub:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
addAddV2add/Const:output:0	mul_2:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_3MulSigmoid:y:0add:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@
SquareSquare7mul_model_spatial_transformation_spatial_layer1_biasadd*
T0*+
_output_shapes
:џџџџџџџџџ@^
mul_4Mulresult_grads_0
Square:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?e
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*+
_output_shapes
:џџџџџџџџџ@X
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@Z
ConstConst*
_output_shapes
:*
dtype0*!
valueB"          F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: ]
mul_7Mulresult_grads_0	mul_3:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@U
IdentityIdentity	mul_7:z:0*
T0*+
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*\
_input_shapesK
I:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:|
+
_output_shapes
:џџџџџџџџџ@
M
_user_specified_name53model/spatial_transformation/spatial_layer1/BiasAdd:d`

_output_shapes
: 
F
_user_specified_name.,model/spatial_transformation/activation/beta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:[W
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: 
&
 _has_manual_control_dependencies(
+
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0
ѓ	
ё
@__inference_ov2_layer_call_and_return_conditional_losses_9236829

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџ@S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
шЎ
Ў$
 __inference__traced_save_9238246
file_prefix5
#read_disablecopyonread_dense_kernel:@1
#read_1_disablecopyonread_dense_bias:@9
'read_2_disablecopyonread_dense_2_kernel:@3
%read_3_disablecopyonread_dense_2_bias:@9
'read_4_disablecopyonread_dense_1_kernel:@@3
%read_5_disablecopyonread_dense_1_bias:@9
'read_6_disablecopyonread_dense_3_kernel:@@3
%read_7_disablecopyonread_dense_3_bias:@9
'read_8_disablecopyonread_dense_4_kernel:@@3
%read_9_disablecopyonread_dense_4_bias:@;
)read_10_disablecopyonread_output_u_kernel:@5
'read_11_disablecopyonread_output_u_bias:;
)read_12_disablecopyonread_output_v_kernel:@5
'read_13_disablecopyonread_output_v_bias:;
)read_14_disablecopyonread_output_p_kernel:@5
'read_15_disablecopyonread_output_p_bias:A
/read_16_disablecopyonread_spatial_layer1_kernel:@;
-read_17_disablecopyonread_spatial_layer1_bias:@A
/read_18_disablecopyonread_spatial_layer2_kernel:@@;
-read_19_disablecopyonread_spatial_layer2_bias:@Q
;read_20_disablecopyonread_multi_head_attention_query_kernel:@@K
9read_21_disablecopyonread_multi_head_attention_query_bias:@O
9read_22_disablecopyonread_multi_head_attention_key_kernel:@@I
7read_23_disablecopyonread_multi_head_attention_key_bias:@Q
;read_24_disablecopyonread_multi_head_attention_value_kernel:@@K
9read_25_disablecopyonread_multi_head_attention_value_bias:@\
Fread_26_disablecopyonread_multi_head_attention_attention_output_kernel:@@R
Dread_27_disablecopyonread_multi_head_attention_attention_output_bias:@6
$read_28_disablecopyonread_ou1_kernel:@@0
"read_29_disablecopyonread_ou1_bias:@6
$read_30_disablecopyonread_ou2_kernel:@@0
"read_31_disablecopyonread_ou2_bias:@6
$read_32_disablecopyonread_ov1_kernel:@@0
"read_33_disablecopyonread_ov1_bias:@6
$read_34_disablecopyonread_ov2_kernel:@@0
"read_35_disablecopyonread_ov2_bias:@6
$read_36_disablecopyonread_op1_kernel:@@0
"read_37_disablecopyonread_op1_bias:@6
$read_38_disablecopyonread_op2_kernel:@@0
"read_39_disablecopyonread_op2_bias:@
savev2_const
identity_81ЂMergeV2CheckpointsЂRead/DisableCopyOnReadЂRead/ReadVariableOpЂRead_1/DisableCopyOnReadЂRead_1/ReadVariableOpЂRead_10/DisableCopyOnReadЂRead_10/ReadVariableOpЂRead_11/DisableCopyOnReadЂRead_11/ReadVariableOpЂRead_12/DisableCopyOnReadЂRead_12/ReadVariableOpЂRead_13/DisableCopyOnReadЂRead_13/ReadVariableOpЂRead_14/DisableCopyOnReadЂRead_14/ReadVariableOpЂRead_15/DisableCopyOnReadЂRead_15/ReadVariableOpЂRead_16/DisableCopyOnReadЂRead_16/ReadVariableOpЂRead_17/DisableCopyOnReadЂRead_17/ReadVariableOpЂRead_18/DisableCopyOnReadЂRead_18/ReadVariableOpЂRead_19/DisableCopyOnReadЂRead_19/ReadVariableOpЂRead_2/DisableCopyOnReadЂRead_2/ReadVariableOpЂRead_20/DisableCopyOnReadЂRead_20/ReadVariableOpЂRead_21/DisableCopyOnReadЂRead_21/ReadVariableOpЂRead_22/DisableCopyOnReadЂRead_22/ReadVariableOpЂRead_23/DisableCopyOnReadЂRead_23/ReadVariableOpЂRead_24/DisableCopyOnReadЂRead_24/ReadVariableOpЂRead_25/DisableCopyOnReadЂRead_25/ReadVariableOpЂRead_26/DisableCopyOnReadЂRead_26/ReadVariableOpЂRead_27/DisableCopyOnReadЂRead_27/ReadVariableOpЂRead_28/DisableCopyOnReadЂRead_28/ReadVariableOpЂRead_29/DisableCopyOnReadЂRead_29/ReadVariableOpЂRead_3/DisableCopyOnReadЂRead_3/ReadVariableOpЂRead_30/DisableCopyOnReadЂRead_30/ReadVariableOpЂRead_31/DisableCopyOnReadЂRead_31/ReadVariableOpЂRead_32/DisableCopyOnReadЂRead_32/ReadVariableOpЂRead_33/DisableCopyOnReadЂRead_33/ReadVariableOpЂRead_34/DisableCopyOnReadЂRead_34/ReadVariableOpЂRead_35/DisableCopyOnReadЂRead_35/ReadVariableOpЂRead_36/DisableCopyOnReadЂRead_36/ReadVariableOpЂRead_37/DisableCopyOnReadЂRead_37/ReadVariableOpЂRead_38/DisableCopyOnReadЂRead_38/ReadVariableOpЂRead_39/DisableCopyOnReadЂRead_39/ReadVariableOpЂRead_4/DisableCopyOnReadЂRead_4/ReadVariableOpЂRead_5/DisableCopyOnReadЂRead_5/ReadVariableOpЂRead_6/DisableCopyOnReadЂRead_6/ReadVariableOpЂRead_7/DisableCopyOnReadЂRead_7/ReadVariableOpЂRead_8/DisableCopyOnReadЂRead_8/ReadVariableOpЂRead_9/DisableCopyOnReadЂRead_9/ReadVariableOpw
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
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:@w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 
Read_1/ReadVariableOpReadVariableOp#read_1_disablecopyonread_dense_bias^Read_1/DisableCopyOnRead"/device:CPU:0*
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
:@{
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_2_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0m

Identity_4IdentityRead_2/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@c

Identity_5IdentityIdentity_4:output:0"/device:CPU:0*
T0*
_output_shapes

:@y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 Ё
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_2_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
:@{
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_1_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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

:@@y
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 Ё
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_1_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
:@{
Read_6/DisableCopyOnReadDisableCopyOnRead'read_6_disablecopyonread_dense_3_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_6/ReadVariableOpReadVariableOp'read_6_disablecopyonread_dense_3_kernel^Read_6/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0n
Identity_12IdentityRead_6/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_13IdentityIdentity_12:output:0"/device:CPU:0*
T0*
_output_shapes

:@@y
Read_7/DisableCopyOnReadDisableCopyOnRead%read_7_disablecopyonread_dense_3_bias"/device:CPU:0*
_output_shapes
 Ё
Read_7/ReadVariableOpReadVariableOp%read_7_disablecopyonread_dense_3_bias^Read_7/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_14IdentityRead_7/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_15IdentityIdentity_14:output:0"/device:CPU:0*
T0*
_output_shapes
:@{
Read_8/DisableCopyOnReadDisableCopyOnRead'read_8_disablecopyonread_dense_4_kernel"/device:CPU:0*
_output_shapes
 Ї
Read_8/ReadVariableOpReadVariableOp'read_8_disablecopyonread_dense_4_kernel^Read_8/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0n
Identity_16IdentityRead_8/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_17IdentityIdentity_16:output:0"/device:CPU:0*
T0*
_output_shapes

:@@y
Read_9/DisableCopyOnReadDisableCopyOnRead%read_9_disablecopyonread_dense_4_bias"/device:CPU:0*
_output_shapes
 Ё
Read_9/ReadVariableOpReadVariableOp%read_9_disablecopyonread_dense_4_bias^Read_9/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0j
Identity_18IdentityRead_9/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_19IdentityIdentity_18:output:0"/device:CPU:0*
T0*
_output_shapes
:@~
Read_10/DisableCopyOnReadDisableCopyOnRead)read_10_disablecopyonread_output_u_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read_10/ReadVariableOpReadVariableOp)read_10_disablecopyonread_output_u_kernel^Read_10/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_20IdentityRead_10/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_21IdentityIdentity_20:output:0"/device:CPU:0*
T0*
_output_shapes

:@|
Read_11/DisableCopyOnReadDisableCopyOnRead'read_11_disablecopyonread_output_u_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_11/ReadVariableOpReadVariableOp'read_11_disablecopyonread_output_u_bias^Read_11/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_22IdentityRead_11/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_23IdentityIdentity_22:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_12/DisableCopyOnReadDisableCopyOnRead)read_12_disablecopyonread_output_v_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read_12/ReadVariableOpReadVariableOp)read_12_disablecopyonread_output_v_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@|
Read_13/DisableCopyOnReadDisableCopyOnRead'read_13_disablecopyonread_output_v_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_13/ReadVariableOpReadVariableOp'read_13_disablecopyonread_output_v_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:~
Read_14/DisableCopyOnReadDisableCopyOnRead)read_14_disablecopyonread_output_p_kernel"/device:CPU:0*
_output_shapes
 Ћ
Read_14/ReadVariableOpReadVariableOp)read_14_disablecopyonread_output_p_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@|
Read_15/DisableCopyOnReadDisableCopyOnRead'read_15_disablecopyonread_output_p_bias"/device:CPU:0*
_output_shapes
 Ѕ
Read_15/ReadVariableOpReadVariableOp'read_15_disablecopyonread_output_p_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:
Read_16/DisableCopyOnReadDisableCopyOnRead/read_16_disablecopyonread_spatial_layer1_kernel"/device:CPU:0*
_output_shapes
 Б
Read_16/ReadVariableOpReadVariableOp/read_16_disablecopyonread_spatial_layer1_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_32IdentityRead_16/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_33IdentityIdentity_32:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_17/DisableCopyOnReadDisableCopyOnRead-read_17_disablecopyonread_spatial_layer1_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_17/ReadVariableOpReadVariableOp-read_17_disablecopyonread_spatial_layer1_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_34IdentityRead_17/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_35IdentityIdentity_34:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_18/DisableCopyOnReadDisableCopyOnRead/read_18_disablecopyonread_spatial_layer2_kernel"/device:CPU:0*
_output_shapes
 Б
Read_18/ReadVariableOpReadVariableOp/read_18_disablecopyonread_spatial_layer2_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_36IdentityRead_18/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_37IdentityIdentity_36:output:0"/device:CPU:0*
T0*
_output_shapes

:@@
Read_19/DisableCopyOnReadDisableCopyOnRead-read_19_disablecopyonread_spatial_layer2_bias"/device:CPU:0*
_output_shapes
 Ћ
Read_19/ReadVariableOpReadVariableOp-read_19_disablecopyonread_spatial_layer2_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_38IdentityRead_19/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_39IdentityIdentity_38:output:0"/device:CPU:0*
T0*
_output_shapes
:@
Read_20/DisableCopyOnReadDisableCopyOnRead;read_20_disablecopyonread_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 С
Read_20/ReadVariableOpReadVariableOp;read_20_disablecopyonread_multi_head_attention_query_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@
Read_21/DisableCopyOnReadDisableCopyOnRead9read_21_disablecopyonread_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 Л
Read_21/ReadVariableOpReadVariableOp9read_21_disablecopyonread_multi_head_attention_query_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_22/DisableCopyOnReadDisableCopyOnRead9read_22_disablecopyonread_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 П
Read_22/ReadVariableOpReadVariableOp9read_22_disablecopyonread_multi_head_attention_key_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@
Read_23/DisableCopyOnReadDisableCopyOnRead7read_23_disablecopyonread_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 Й
Read_23/ReadVariableOpReadVariableOp7read_23_disablecopyonread_multi_head_attention_key_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_24/DisableCopyOnReadDisableCopyOnRead;read_24_disablecopyonread_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 С
Read_24/ReadVariableOpReadVariableOp;read_24_disablecopyonread_multi_head_attention_value_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_48IdentityRead_24/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_49IdentityIdentity_48:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@
Read_25/DisableCopyOnReadDisableCopyOnRead9read_25_disablecopyonread_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 Л
Read_25/ReadVariableOpReadVariableOp9read_25_disablecopyonread_multi_head_attention_value_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_50IdentityRead_25/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_51IdentityIdentity_50:output:0"/device:CPU:0*
T0*
_output_shapes

:@
Read_26/DisableCopyOnReadDisableCopyOnReadFread_26_disablecopyonread_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 Ь
Read_26/ReadVariableOpReadVariableOpFread_26_disablecopyonread_multi_head_attention_attention_output_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@
Read_27/DisableCopyOnReadDisableCopyOnReadDread_27_disablecopyonread_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 Т
Read_27/ReadVariableOpReadVariableOpDread_27_disablecopyonread_multi_head_attention_attention_output_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_28/DisableCopyOnReadDisableCopyOnRead$read_28_disablecopyonread_ou1_kernel"/device:CPU:0*
_output_shapes
 І
Read_28/ReadVariableOpReadVariableOp$read_28_disablecopyonread_ou1_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_29/DisableCopyOnReadDisableCopyOnRead"read_29_disablecopyonread_ou1_bias"/device:CPU:0*
_output_shapes
  
Read_29/ReadVariableOpReadVariableOp"read_29_disablecopyonread_ou1_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_30/DisableCopyOnReadDisableCopyOnRead$read_30_disablecopyonread_ou2_kernel"/device:CPU:0*
_output_shapes
 І
Read_30/ReadVariableOpReadVariableOp$read_30_disablecopyonread_ou2_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_31/DisableCopyOnReadDisableCopyOnRead"read_31_disablecopyonread_ou2_bias"/device:CPU:0*
_output_shapes
  
Read_31/ReadVariableOpReadVariableOp"read_31_disablecopyonread_ou2_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_62IdentityRead_31/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_63IdentityIdentity_62:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_32/DisableCopyOnReadDisableCopyOnRead$read_32_disablecopyonread_ov1_kernel"/device:CPU:0*
_output_shapes
 І
Read_32/ReadVariableOpReadVariableOp$read_32_disablecopyonread_ov1_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_33/DisableCopyOnReadDisableCopyOnRead"read_33_disablecopyonread_ov1_bias"/device:CPU:0*
_output_shapes
  
Read_33/ReadVariableOpReadVariableOp"read_33_disablecopyonread_ov1_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_34/DisableCopyOnReadDisableCopyOnRead$read_34_disablecopyonread_ov2_kernel"/device:CPU:0*
_output_shapes
 І
Read_34/ReadVariableOpReadVariableOp$read_34_disablecopyonread_ov2_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_35/DisableCopyOnReadDisableCopyOnRead"read_35_disablecopyonread_ov2_bias"/device:CPU:0*
_output_shapes
  
Read_35/ReadVariableOpReadVariableOp"read_35_disablecopyonread_ov2_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_36/DisableCopyOnReadDisableCopyOnRead$read_36_disablecopyonread_op1_kernel"/device:CPU:0*
_output_shapes
 І
Read_36/ReadVariableOpReadVariableOp$read_36_disablecopyonread_op1_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_37/DisableCopyOnReadDisableCopyOnRead"read_37_disablecopyonread_op1_bias"/device:CPU:0*
_output_shapes
  
Read_37/ReadVariableOpReadVariableOp"read_37_disablecopyonread_op1_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_38/DisableCopyOnReadDisableCopyOnRead$read_38_disablecopyonread_op2_kernel"/device:CPU:0*
_output_shapes
 І
Read_38/ReadVariableOpReadVariableOp$read_38_disablecopyonread_op2_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_39/DisableCopyOnReadDisableCopyOnRead"read_39_disablecopyonread_op2_bias"/device:CPU:0*
_output_shapes
  
Read_39/ReadVariableOpReadVariableOp"read_39_disablecopyonread_op2_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_78IdentityRead_39/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_79IdentityIdentity_78:output:0"/device:CPU:0*
T0*
_output_shapes
:@д
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*§
valueѓB№)B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-11/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-11/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/22/.ATTRIBUTES/VARIABLE_VALUEB'variables/23/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/32/.ATTRIBUTES/VARIABLE_VALUEB'variables/33/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPHП
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:)*
dtype0*e
value\BZ)B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B ы
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *7
dtypes-
+2)
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:Г
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 i
Identity_80Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: U
Identity_81IdentityIdentity_80:output:0^NoOp*
T0*
_output_shapes
: ч
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "#
identity_81Identity_81:output:0*(
_construction_contextkEagerRuntime*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_26/ReadVariableOpRead_26/ReadVariableOp26
Read_27/DisableCopyOnReadRead_27/DisableCopyOnRead20
Read_27/ReadVariableOpRead_27/ReadVariableOp26
Read_28/DisableCopyOnReadRead_28/DisableCopyOnRead20
Read_28/ReadVariableOpRead_28/ReadVariableOp26
Read_29/DisableCopyOnReadRead_29/DisableCopyOnRead20
Read_29/ReadVariableOpRead_29/ReadVariableOp24
Read_3/DisableCopyOnReadRead_3/DisableCopyOnRead2.
Read_3/ReadVariableOpRead_3/ReadVariableOp26
Read_30/DisableCopyOnReadRead_30/DisableCopyOnRead20
Read_30/ReadVariableOpRead_30/ReadVariableOp26
Read_31/DisableCopyOnReadRead_31/DisableCopyOnRead20
Read_31/ReadVariableOpRead_31/ReadVariableOp26
Read_32/DisableCopyOnReadRead_32/DisableCopyOnRead20
Read_32/ReadVariableOpRead_32/ReadVariableOp26
Read_33/DisableCopyOnReadRead_33/DisableCopyOnRead20
Read_33/ReadVariableOpRead_33/ReadVariableOp26
Read_34/DisableCopyOnReadRead_34/DisableCopyOnRead20
Read_34/ReadVariableOpRead_34/ReadVariableOp26
Read_35/DisableCopyOnReadRead_35/DisableCopyOnRead20
Read_35/ReadVariableOpRead_35/ReadVariableOp26
Read_36/DisableCopyOnReadRead_36/DisableCopyOnRead20
Read_36/ReadVariableOpRead_36/ReadVariableOp26
Read_37/DisableCopyOnReadRead_37/DisableCopyOnRead20
Read_37/ReadVariableOpRead_37/ReadVariableOp26
Read_38/DisableCopyOnReadRead_38/DisableCopyOnRead20
Read_38/ReadVariableOpRead_38/ReadVariableOp26
Read_39/DisableCopyOnReadRead_39/DisableCopyOnRead20
Read_39/ReadVariableOpRead_39/ReadVariableOp24
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
Read_9/ReadVariableOpRead_9/ReadVariableOp:=)9

_output_shapes
: 

_user_specified_nameConst:(($
"
_user_specified_name
op2/bias:*'&
$
_user_specified_name
op2/kernel:(&$
"
_user_specified_name
op1/bias:*%&
$
_user_specified_name
op1/kernel:($$
"
_user_specified_name
ov2/bias:*#&
$
_user_specified_name
ov2/kernel:("$
"
_user_specified_name
ov1/bias:*!&
$
_user_specified_name
ov1/kernel:( $
"
_user_specified_name
ou2/bias:*&
$
_user_specified_name
ou2/kernel:($
"
_user_specified_name
ou1/bias:*&
$
_user_specified_name
ou1/kernel:JF
D
_user_specified_name,*multi_head_attention/attention_output/bias:LH
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?;
9
_user_specified_name!multi_head_attention/value/bias:A=
;
_user_specified_name#!multi_head_attention/value/kernel:=9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:3/
-
_user_specified_namespatial_layer2/bias:51
/
_user_specified_namespatial_layer2/kernel:3/
-
_user_specified_namespatial_layer1/bias:51
/
_user_specified_namespatial_layer1/kernel:-)
'
_user_specified_nameoutput_p/bias:/+
)
_user_specified_nameoutput_p/kernel:-)
'
_user_specified_nameoutput_v/bias:/+
)
_user_specified_nameoutput_v/kernel:-)
'
_user_specified_nameoutput_u/bias:/+
)
_user_specified_nameoutput_u/kernel:,
(
&
_user_specified_namedense_4/bias:.	*
(
_user_specified_namedense_4/kernel:,(
&
_user_specified_namedense_3/bias:.*
(
_user_specified_namedense_3/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:,(
&
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:*&
$
_user_specified_name
dense/bias:,(
&
_user_specified_namedense/kernel:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ї

)__inference_dense_2_layer_call_fn_9236204

inputs
unknown:@
	unknown_0:@
identityЂStatefulPartitionedCallщ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_9235099|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	9236200:'#
!
_user_specified_name	9236198:\ X
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
#
§
D__inference_dense_2_layer_call_and_return_conditional_losses_9236243

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@

identity_1ЂBiasAdd/ReadVariableOpЂTensordot/ReadVariableOpz
Tensordot/ReadVariableOpReadVariableOp!tensordot_readvariableop_resource*
_output_shapes

:@*
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
::эЯY
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : Л
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
value	B : П
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
value	B : 
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ@[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : Ї
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@I
betaConst*
_output_shapes
: *
dtype0*
valueB
 *  ?j
mulMulbeta:output:0BiasAdd:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@Z
SigmoidSigmoidmul:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@j
mul_1MulBiasAdd:output:0Sigmoid:y:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@^
IdentityIdentity	mul_1:z:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@з
	IdentityN	IdentityN	mul_1:z:0BiasAdd:output:0beta:output:0*
T
2*-
_gradient_op_typeCustomGradient-9236234*V
_output_shapesD
B:џџџџџџџџџџџџџџџџџџ@:џџџџџџџџџџџџџџџџџџ@: p

Identity_1IdentityIdentityN:output:0^NoOp*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ@V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:џџџџџџџџџџџџџџџџџџ: : 20
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
 :џџџџџџџџџџџџџџџџџџ
 
_user_specified_nameinputs
ј	
і
E__inference_output_p_layer_call_and_return_conditional_losses_9236585

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityЂBiasAdd/ReadVariableOpЂMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџr
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:џџџџџџџџџ_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:џџџџџџџџџS
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:џџџџџџџџџ@: : 20
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
:џџџџџџџџџ@
 
_user_specified_nameinputs
Ц

$__inference_internal_grad_fn_9237338
result_grads_0
result_grads_1
result_grads_2
mul_beta

mul_inputs
identity

identity_1c
mulMulmul_beta
mul_inputs^result_grads_0*
T0*'
_output_shapes
:џџџџџџџџџ@M
SigmoidSigmoidmul:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_1Mulmul_beta
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@N
	sub/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
subSubsub/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@R
mul_2Mul	mul_1:z:0sub:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
	add/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?]
addAddV2add/Const:output:0	mul_2:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_3MulSigmoid:y:0add:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@N
SquareSquare
mul_inputs*
T0*'
_output_shapes
:џџџџџџџџџ@Z
mul_4Mulresult_grads_0
Square:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
mul_5Mul	mul_4:z:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@P
sub_1/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ?a
sub_1Subsub_1/Const:output:0Sigmoid:y:0*
T0*'
_output_shapes
:џџџџџџџџџ@T
mul_6Mul	mul_5:z:0	sub_1:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@V
ConstConst*
_output_shapes
:*
dtype0*
valueB"       F
SumSum	mul_6:z:0Const:output:0*
T0*
_output_shapes
: Y
mul_7Mulresult_grads_0	mul_3:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@Q
IdentityIdentity	mul_7:z:0*
T0*'
_output_shapes
:џџџџџџџџџ@E

Identity_1IdentitySum:output:0*
T0*
_output_shapes
: "!

identity_1Identity_1:output:0"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*P
_input_shapes?
=:џџџџџџџџџ@:џџџџџџџџџ@: : :џџџџџџџџџ@:OK
'
_output_shapes
:џџџџџџџџџ@
 
_user_specified_nameinputs:<8

_output_shapes
: 

_user_specified_namebeta:FB

_output_shapes
: 
(
_user_specified_nameresult_grads_2:WS
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_1: {
&
 _has_manual_control_dependencies(
'
_output_shapes
:џџџџџџџџџ@
(
_user_specified_nameresult_grads_0

i
/__inference_concatenate_2_layer_call_fn_9236121
inputs_0
inputs_1
inputs_2
identityн
PartitionedCallPartitionedCallinputs_0inputs_1inputs_2*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9235041m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*s
_input_shapesb
`:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:џџџџџџџџџџџџџџџџџџ:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_2:^Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
"
_user_specified_name
inputs_0>
$__inference_internal_grad_fn_9237041CustomGradient-9236912>
$__inference_internal_grad_fn_9237068CustomGradient-9234919>
$__inference_internal_grad_fn_9237095CustomGradient-9236875>
$__inference_internal_grad_fn_9237122CustomGradient-9234890>
$__inference_internal_grad_fn_9237149CustomGradient-9236838>
$__inference_internal_grad_fn_9237176CustomGradient-9234787>
$__inference_internal_grad_fn_9237203CustomGradient-9236801>
$__inference_internal_grad_fn_9237230CustomGradient-9234758>
$__inference_internal_grad_fn_9237257CustomGradient-9236764>
$__inference_internal_grad_fn_9237284CustomGradient-9234655>
$__inference_internal_grad_fn_9237311CustomGradient-9236727>
$__inference_internal_grad_fn_9237338CustomGradient-9234626>
$__inference_internal_grad_fn_9237365CustomGradient-9236690>
$__inference_internal_grad_fn_9237392CustomGradient-9234523>
$__inference_internal_grad_fn_9237419CustomGradient-9236633>
$__inference_internal_grad_fn_9237446CustomGradient-9234474>
$__inference_internal_grad_fn_9237473CustomGradient-9236508>
$__inference_internal_grad_fn_9237500CustomGradient-9235349>
$__inference_internal_grad_fn_9237527CustomGradient-9236330>
$__inference_internal_grad_fn_9237554CustomGradient-9235245>
$__inference_internal_grad_fn_9237581CustomGradient-9236282>
$__inference_internal_grad_fn_9237608CustomGradient-9235201>
$__inference_internal_grad_fn_9237635CustomGradient-9236234>
$__inference_internal_grad_fn_9237662CustomGradient-9235090>
$__inference_internal_grad_fn_9237689CustomGradient-9236186>
$__inference_internal_grad_fn_9237716CustomGradient-9235134>
$__inference_internal_grad_fn_9237743CustomGradient-9234062>
$__inference_internal_grad_fn_9237770CustomGradient-9234097>
$__inference_internal_grad_fn_9237797CustomGradient-9234141>
$__inference_internal_grad_fn_9237824CustomGradient-9234176>
$__inference_internal_grad_fn_9237851CustomGradient-9234211>
$__inference_internal_grad_fn_9237878CustomGradient-9234246>
$__inference_internal_grad_fn_9237905CustomGradient-9234312>
$__inference_internal_grad_fn_9237932CustomGradient-9234329>
$__inference_internal_grad_fn_9237959CustomGradient-9234344>
$__inference_internal_grad_fn_9237986CustomGradient-9234359>
$__inference_internal_grad_fn_9238013CustomGradient-9234374>
$__inference_internal_grad_fn_9238040CustomGradient-9234389>
$__inference_internal_grad_fn_9238067CustomGradient-9234404"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ѕ
serving_defaultс
L
	Pbc_layer?
serving_default_Pbc_layer:0џџџџџџџџџџџџџџџџџџ
;
T_layer0
serving_default_T_layer:0џџџџџџџџџ
L
	Tbc_layer?
serving_default_Tbc_layer:0џџџџџџџџџџџџџџџџџџ
L
	Ubc_layer?
serving_default_Ubc_layer:0џџџџџџџџџџџџџџџџџџ
L
	Vbc_layer?
serving_default_Vbc_layer:0џџџџџџџџџџџџџџџџџџ
;
X_layer0
serving_default_X_layer:0џџџџџџџџџ
L
	Xbc_layer?
serving_default_Xbc_layer:0џџџџџџџџџџџџџџџџџџ
;
Y_layer0
serving_default_Y_layer:0џџџџџџџџџ
L
	Ybc_layer?
serving_default_Ybc_layer:0џџџџџџџџџџџџџџџџџџ<
output_p0
StatefulPartitionedCall:0џџџџџџџџџ<
output_u0
StatefulPartitionedCall:1џџџџџџџџџ<
output_v0
StatefulPartitionedCall:2џџџџџџџџџtensorflow/serving/predict:ЖЈ
У
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
layer-12
layer-13
layer-14
layer_with_weights-0
layer-15
layer_with_weights-1
layer-16
layer_with_weights-2
layer-17
layer_with_weights-3
layer-18
layer_with_weights-4
layer-19
layer_with_weights-5
layer-20
layer-21
layer_with_weights-6
layer-22
layer-23
layer_with_weights-7
layer-24
layer_with_weights-8
layer-25
layer_with_weights-9
layer-26
layer_with_weights-10
layer-27
layer_with_weights-11
layer-28
layer_with_weights-12
layer-29
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses
%_default_save_signature
&
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ѕ
'	variables
(trainable_variables
)regularization_losses
*	keras_api
+__call__
*,&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ѕ
-	variables
.trainable_variables
/regularization_losses
0	keras_api
1__call__
*2&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
Ѕ
3	variables
4trainable_variables
5regularization_losses
6	keras_api
7__call__
*8&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
9	variables
:trainable_variables
;regularization_losses
<	keras_api
=__call__
*>&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
?	variables
@trainable_variables
Aregularization_losses
B	keras_api
C__call__
*D&call_and_return_all_conditional_losses"
_tf_keras_layer
Ѕ
E	variables
Ftrainable_variables
Gregularization_losses
H	keras_api
I__call__
*J&call_and_return_all_conditional_losses"
_tf_keras_layer
Л
K	variables
Ltrainable_variables
Mregularization_losses
N	keras_api
O__call__
*P&call_and_return_all_conditional_losses

Qkernel
Rbias"
_tf_keras_layer
Л
S	variables
Ttrainable_variables
Uregularization_losses
V	keras_api
W__call__
*X&call_and_return_all_conditional_losses

Ykernel
Zbias"
_tf_keras_layer

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
Л
e	variables
ftrainable_variables
gregularization_losses
h	keras_api
i__call__
*j&call_and_return_all_conditional_losses

kkernel
lbias"
_tf_keras_layer
Л
m	variables
ntrainable_variables
oregularization_losses
p	keras_api
q__call__
*r&call_and_return_all_conditional_losses

skernel
tbias"
_tf_keras_layer

u	variables
vtrainable_variables
wregularization_losses
x	keras_api
y__call__
*z&call_and_return_all_conditional_losses
{_query_dense
|
_key_dense
}_value_dense
~_softmax
_dropout_layer
_output_dense"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer
У
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
kernel
	bias"
_tf_keras_layer
Ћ
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_layer

layer_with_weights-0
layer-0
layer-1
layer_with_weights-1
layer-2
layer-3
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"
_tf_keras_sequential

layer_with_weights-0
layer-0
 layer-1
Ёlayer_with_weights-1
Ёlayer-2
Ђlayer-3
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
І	keras_api
Ї__call__
+Ј&call_and_return_all_conditional_losses"
_tf_keras_sequential

Љlayer_with_weights-0
Љlayer-0
Њlayer-1
Ћlayer_with_weights-1
Ћlayer-2
Ќlayer-3
­	variables
Ўtrainable_variables
Џregularization_losses
А	keras_api
Б__call__
+В&call_and_return_all_conditional_losses"
_tf_keras_sequential
У
Г	variables
Дtrainable_variables
Еregularization_losses
Ж	keras_api
З__call__
+И&call_and_return_all_conditional_losses
Йkernel
	Кbias"
_tf_keras_layer
У
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses
Сkernel
	Тbias"
_tf_keras_layer
У
У	variables
Фtrainable_variables
Хregularization_losses
Ц	keras_api
Ч__call__
+Ш&call_and_return_all_conditional_losses
Щkernel
	Ъbias"
_tf_keras_layer
і
Q0
R1
Y2
Z3
Ы4
Ь5
Э6
Ю7
k8
l9
s10
t11
Я12
а13
б14
в15
г16
д17
е18
ж19
20
21
з22
и23
й24
к25
л26
м27
н28
о29
п30
р31
с32
т33
Й34
К35
С36
Т37
Щ38
Ъ39"
trackable_list_wrapper
і
Q0
R1
Y2
Z3
Ы4
Ь5
Э6
Ю7
k8
l9
s10
t11
Я12
а13
б14
в15
г16
д17
е18
ж19
20
21
з22
и23
й24
к25
л26
м27
н28
о29
п30
р31
с32
т33
Й34
К35
С36
Т37
Щ38
Ъ39"
trackable_list_wrapper
 "
trackable_list_wrapper
Я
уnon_trainable_variables
фlayers
хmetrics
 цlayer_regularization_losses
чlayer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
%_default_save_signature
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
Х
шtrace_0
щtrace_12
'__inference_model_layer_call_fn_9235694
'__inference_model_layer_call_fn_9235791Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zшtrace_0zщtrace_1
ћ
ъtrace_0
ыtrace_12Р
B__inference_model_layer_call_and_return_conditional_losses_9235446
B__inference_model_layer_call_and_return_conditional_losses_9235597Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zъtrace_0zыtrace_1
ЁB
"__inference__wrapped_model_9234433X_layerY_layerT_layer	Xbc_layer	Ybc_layer	Tbc_layer	Ubc_layer	Vbc_layer	Pbc_layer	"
В
FullArgSpec
args

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
-
ьserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
'	variables
(trainable_variables
)regularization_losses
+__call__
*,&call_and_return_all_conditional_losses
&,"call_and_return_conditional_losses"
_generic_user_object
ч
ђtrace_02Ш
+__inference_rescaling_layer_call_fn_9236059
В
FullArgSpec
args

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
annotationsЊ *
 zђtrace_0

ѓtrace_02у
F__inference_rescaling_layer_call_and_return_conditional_losses_9236069
В
FullArgSpec
args

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
annotationsЊ *
 zѓtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
-	variables
.trainable_variables
/regularization_losses
1__call__
*2&call_and_return_all_conditional_losses
&2"call_and_return_conditional_losses"
_generic_user_object
щ
љtrace_02Ъ
-__inference_rescaling_1_layer_call_fn_9236074
В
FullArgSpec
args

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
annotationsЊ *
 zљtrace_0

њtrace_02х
H__inference_rescaling_1_layer_call_and_return_conditional_losses_9236084
В
FullArgSpec
args

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
annotationsЊ *
 zњtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
3	variables
4trainable_variables
5regularization_losses
7__call__
*8&call_and_return_all_conditional_losses
&8"call_and_return_conditional_losses"
_generic_user_object
щ
trace_02Ъ
-__inference_concatenate_layer_call_fn_9236091
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02х
H__inference_concatenate_layer_call_and_return_conditional_losses_9236099
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
9	variables
:trainable_variables
;regularization_losses
=__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
/__inference_concatenate_1_layer_call_fn_9236106
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02ч
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9236114
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
?	variables
@trainable_variables
Aregularization_losses
C__call__
*D&call_and_return_all_conditional_losses
&D"call_and_return_conditional_losses"
_generic_user_object
ы
trace_02Ь
/__inference_concatenate_2_layer_call_fn_9236121
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02ч
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9236129
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
E	variables
Ftrainable_variables
Gregularization_losses
I__call__
*J&call_and_return_all_conditional_losses
&J"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_reshape_layer_call_fn_9236134
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02с
D__inference_reshape_layer_call_and_return_conditional_losses_9236147
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
.
Q0
R1"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
K	variables
Ltrainable_variables
Mregularization_losses
O__call__
*P&call_and_return_all_conditional_losses
&P"call_and_return_conditional_losses"
_generic_user_object
у
trace_02Ф
'__inference_dense_layer_call_fn_9236156
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
ў
trace_02п
B__inference_dense_layer_call_and_return_conditional_losses_9236195
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
:@2dense/kernel
:@2
dense/bias
.
Y0
Z1"
trackable_list_wrapper
.
Y0
Z1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
S	variables
Ttrainable_variables
Uregularization_losses
W__call__
*X&call_and_return_all_conditional_losses
&X"call_and_return_conditional_losses"
_generic_user_object
х
Ѓtrace_02Ц
)__inference_dense_2_layer_call_fn_9236204
В
FullArgSpec
args

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
annotationsЊ *
 zЃtrace_0

Єtrace_02с
D__inference_dense_2_layer_call_and_return_conditional_losses_9236243
В
FullArgSpec
args

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
annotationsЊ *
 zЄtrace_0
 :@2dense_2/kernel
:@2dense_2/bias
У
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses
Ыkernel
	Ьbias"
_tf_keras_layer
Ћ
Ћ	variables
Ќtrainable_variables
­regularization_losses
Ў	keras_api
Џ__call__
+А&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses
Эkernel
	Юbias"
_tf_keras_layer
Ћ
З	variables
Иtrainable_variables
Йregularization_losses
К	keras_api
Л__call__
+М&call_and_return_all_conditional_losses"
_tf_keras_layer
@
Ы0
Ь1
Э2
Ю3"
trackable_list_wrapper
@
Ы0
Ь1
Э2
Ю3"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
_	variables
`trainable_variables
aregularization_losses
c__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
ч
Тtrace_0
Уtrace_12Ќ
8__inference_spatial_transformation_layer_call_fn_9234564
8__inference_spatial_transformation_layer_call_fn_9234577Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zТtrace_0zУtrace_1

Фtrace_0
Хtrace_12т
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234535
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234551Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zФtrace_0zХtrace_1
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
В
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
e	variables
ftrainable_variables
gregularization_losses
i__call__
*j&call_and_return_all_conditional_losses
&j"call_and_return_conditional_losses"
_generic_user_object
х
Ыtrace_02Ц
)__inference_dense_1_layer_call_fn_9236252
В
FullArgSpec
args

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
annotationsЊ *
 zЫtrace_0

Ьtrace_02с
D__inference_dense_1_layer_call_and_return_conditional_losses_9236291
В
FullArgSpec
args

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
annotationsЊ *
 zЬtrace_0
 :@@2dense_1/kernel
:@2dense_1/bias
.
s0
t1"
trackable_list_wrapper
.
s0
t1"
trackable_list_wrapper
 "
trackable_list_wrapper
В
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
m	variables
ntrainable_variables
oregularization_losses
q__call__
*r&call_and_return_all_conditional_losses
&r"call_and_return_conditional_losses"
_generic_user_object
х
вtrace_02Ц
)__inference_dense_3_layer_call_fn_9236300
В
FullArgSpec
args

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
annotationsЊ *
 zвtrace_0

гtrace_02с
D__inference_dense_3_layer_call_and_return_conditional_losses_9236339
В
FullArgSpec
args

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
annotationsЊ *
 zгtrace_0
 :@@2dense_3/kernel
:@2dense_3/bias
`
Я0
а1
б2
в3
г4
д5
е6
ж7"
trackable_list_wrapper
`
Я0
а1
б2
в3
г4
д5
е6
ж7"
trackable_list_wrapper
 "
trackable_list_wrapper
В
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
u	variables
vtrainable_variables
wregularization_losses
y__call__
*z&call_and_return_all_conditional_losses
&z"call_and_return_conditional_losses"
_generic_user_object
Ж
йtrace_0
кtrace_12ћ
6__inference_multi_head_attention_layer_call_fn_9236362
6__inference_multi_head_attention_layer_call_fn_9236385
В§
FullArgSpecp
argshe
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
defaultsЂ

 

 
p 
p 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zйtrace_0zкtrace_1
ь
лtrace_0
мtrace_12Б
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9236421
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9236457
В§
FullArgSpecp
argshe
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
defaultsЂ

 

 
p 
p 
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zлtrace_0zмtrace_1
і
н	variables
оtrainable_variables
пregularization_losses
р	keras_api
с__call__
+т&call_and_return_all_conditional_losses
уpartial_output_shape
фfull_output_shape
Яkernel
	аbias"
_tf_keras_layer
і
х	variables
цtrainable_variables
чregularization_losses
ш	keras_api
щ__call__
+ъ&call_and_return_all_conditional_losses
ыpartial_output_shape
ьfull_output_shape
бkernel
	вbias"
_tf_keras_layer
і
э	variables
юtrainable_variables
яregularization_losses
№	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses
ѓpartial_output_shape
єfull_output_shape
гkernel
	дbias"
_tf_keras_layer
Ћ
ѕ	variables
іtrainable_variables
їregularization_losses
ј	keras_api
љ__call__
+њ&call_and_return_all_conditional_losses"
_tf_keras_layer
У
ћ	variables
ќtrainable_variables
§regularization_losses
ў	keras_api
џ__call__
+&call_and_return_all_conditional_losses
_random_generator"
_tf_keras_layer
і
	variables
trainable_variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses
partial_output_shape
full_output_shape
еkernel
	жbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
с
trace_02Т
%__inference_add_layer_call_fn_9236463
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
ќ
trace_02н
@__inference_add_layer_call_and_return_conditional_losses_9236469
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
0
0
1"
trackable_list_wrapper
0
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_dense_4_layer_call_fn_9236478
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02с
D__inference_dense_4_layer_call_and_return_conditional_losses_9236517
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 :@@2dense_4/kernel
:@2dense_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
х
trace_02Ц
)__inference_flatten_layer_call_fn_9236522
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02с
D__inference_flatten_layer_call_and_return_conditional_losses_9236528
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
У
	variables
 trainable_variables
Ёregularization_losses
Ђ	keras_api
Ѓ__call__
+Є&call_and_return_all_conditional_losses
зkernel
	иbias"
_tf_keras_layer
Ћ
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Ј	keras_api
Љ__call__
+Њ&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Ћ	variables
Ќtrainable_variables
­regularization_losses
Ў	keras_api
Џ__call__
+А&call_and_return_all_conditional_losses
йkernel
	кbias"
_tf_keras_layer
Ћ
Б	variables
Вtrainable_variables
Гregularization_losses
Д	keras_api
Е__call__
+Ж&call_and_return_all_conditional_losses"
_tf_keras_layer
@
з0
и1
й2
к3"
trackable_list_wrapper
@
з0
и1
й2
к3"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Н
Мtrace_0
Нtrace_12
#__inference_U_layer_call_fn_9234696
#__inference_U_layer_call_fn_9234709Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zМtrace_0zНtrace_1
ѓ
Оtrace_0
Пtrace_12И
>__inference_U_layer_call_and_return_conditional_losses_9234667
>__inference_U_layer_call_and_return_conditional_losses_9234683Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zОtrace_0zПtrace_1
У
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses
лkernel
	мbias"
_tf_keras_layer
Ћ
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
У
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses
нkernel
	оbias"
_tf_keras_layer
Ћ
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses"
_tf_keras_layer
@
л0
м1
н2
о3"
trackable_list_wrapper
@
л0
м1
н2
о3"
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ѓ	variables
Єtrainable_variables
Ѕregularization_losses
Ї__call__
+Ј&call_and_return_all_conditional_losses
'Ј"call_and_return_conditional_losses"
_generic_user_object
Н
нtrace_0
оtrace_12
#__inference_V_layer_call_fn_9234828
#__inference_V_layer_call_fn_9234841Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zнtrace_0zоtrace_1
ѓ
пtrace_0
рtrace_12И
>__inference_V_layer_call_and_return_conditional_losses_9234799
>__inference_V_layer_call_and_return_conditional_losses_9234815Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zпtrace_0zрtrace_1
У
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses
пkernel
	рbias"
_tf_keras_layer
Ћ
ч	variables
шtrainable_variables
щregularization_losses
ъ	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses"
_tf_keras_layer
У
э	variables
юtrainable_variables
яregularization_losses
№	keras_api
ё__call__
+ђ&call_and_return_all_conditional_losses
сkernel
	тbias"
_tf_keras_layer
Ћ
ѓ	variables
єtrainable_variables
ѕregularization_losses
і	keras_api
ї__call__
+ј&call_and_return_all_conditional_losses"
_tf_keras_layer
@
п0
р1
с2
т3"
trackable_list_wrapper
@
п0
р1
с2
т3"
trackable_list_wrapper
 "
trackable_list_wrapper
И
љnon_trainable_variables
њlayers
ћmetrics
 ќlayer_regularization_losses
§layer_metrics
­	variables
Ўtrainable_variables
Џregularization_losses
Б__call__
+В&call_and_return_all_conditional_losses
'В"call_and_return_conditional_losses"
_generic_user_object
Н
ўtrace_0
џtrace_12
#__inference_P_layer_call_fn_9234960
#__inference_P_layer_call_fn_9234973Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 zўtrace_0zџtrace_1
ѓ
trace_0
trace_12И
>__inference_P_layer_call_and_return_conditional_losses_9234931
>__inference_P_layer_call_and_return_conditional_losses_9234947Е
ЎВЊ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsЂ
p 

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 ztrace_0ztrace_1
0
Й0
К1"
trackable_list_wrapper
0
Й0
К1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Г	variables
Дtrainable_variables
Еregularization_losses
З__call__
+И&call_and_return_all_conditional_losses
'И"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_output_u_layer_call_fn_9236537
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02т
E__inference_output_u_layer_call_and_return_conditional_losses_9236547
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
!:@2output_u/kernel
:2output_u/bias
0
С0
Т1"
trackable_list_wrapper
0
С0
Т1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_output_v_layer_call_fn_9236556
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02т
E__inference_output_v_layer_call_and_return_conditional_losses_9236566
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
!:@2output_v/kernel
:2output_v/bias
0
Щ0
Ъ1"
trackable_list_wrapper
0
Щ0
Ъ1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
У	variables
Фtrainable_variables
Хregularization_losses
Ч__call__
+Ш&call_and_return_all_conditional_losses
'Ш"call_and_return_conditional_losses"
_generic_user_object
ц
trace_02Ч
*__inference_output_p_layer_call_fn_9236575
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02т
E__inference_output_p_layer_call_and_return_conditional_losses_9236585
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
!:@2output_p/kernel
:2output_p/bias
':%@2spatial_layer1/kernel
!:@2spatial_layer1/bias
':%@@2spatial_layer2/kernel
!:@2spatial_layer2/bias
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
:@@2
ov1/kernel
:@2ov1/bias
:@@2
ov2/kernel
:@2ov2/bias
:@@2
op1/kernel
:@2op1/bias
:@@2
op2/kernel
:@2op2/bias
 "
trackable_list_wrapper

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
21
22
23
24
25
26
27
28
29"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
КBЗ
'__inference_model_layer_call_fn_9235694X_layerY_layerT_layer	Xbc_layer	Ybc_layer	Tbc_layer	Ubc_layer	Vbc_layer	Pbc_layer	"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
КBЗ
'__inference_model_layer_call_fn_9235791X_layerY_layerT_layer	Xbc_layer	Ybc_layer	Tbc_layer	Ubc_layer	Vbc_layer	Pbc_layer	"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
еBв
B__inference_model_layer_call_and_return_conditional_losses_9235446X_layerY_layerT_layer	Xbc_layer	Ybc_layer	Tbc_layer	Ubc_layer	Vbc_layer	Pbc_layer	"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
еBв
B__inference_model_layer_call_and_return_conditional_losses_9235597X_layerY_layerT_layer	Xbc_layer	Ybc_layer	Tbc_layer	Ubc_layer	Vbc_layer	Pbc_layer	"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
%__inference_signature_wrapper_9236054	Pbc_layerT_layer	Tbc_layer	Ubc_layer	Vbc_layerX_layer	Xbc_layerY_layer	Ybc_layer"ў
їВѓ
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargsro
j	Pbc_layer
	jT_layer
j	Tbc_layer
j	Ubc_layer
j	Vbc_layer
	jX_layer
j	Xbc_layer
	jY_layer
j	Ybc_layer
kwonlydefaults
 
annotationsЊ *
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
еBв
+__inference_rescaling_layer_call_fn_9236059inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
№Bэ
F__inference_rescaling_layer_call_and_return_conditional_losses_9236069inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
зBд
-__inference_rescaling_1_layer_call_fn_9236074inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ђBя
H__inference_rescaling_1_layer_call_and_return_conditional_losses_9236084inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
эBъ
-__inference_concatenate_layer_call_fn_9236091inputs_0inputs_1inputs_2"
В
FullArgSpec
args

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
annotationsЊ *
 
B
H__inference_concatenate_layer_call_and_return_conditional_losses_9236099inputs_0inputs_1inputs_2"
В
FullArgSpec
args

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
annotationsЊ *
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
яBь
/__inference_concatenate_1_layer_call_fn_9236106inputs_0inputs_1inputs_2"
В
FullArgSpec
args

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
annotationsЊ *
 
B
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9236114inputs_0inputs_1inputs_2"
В
FullArgSpec
args

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
annotationsЊ *
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
яBь
/__inference_concatenate_2_layer_call_fn_9236121inputs_0inputs_1inputs_2"
В
FullArgSpec
args

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
annotationsЊ *
 
B
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9236129inputs_0inputs_1inputs_2"
В
FullArgSpec
args

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
annotationsЊ *
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
гBа
)__inference_reshape_layer_call_fn_9236134inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_reshape_layer_call_and_return_conditional_losses_9236147inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
бBЮ
'__inference_dense_layer_call_fn_9236156inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ьBщ
B__inference_dense_layer_call_and_return_conditional_losses_9236195inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
гBа
)__inference_dense_2_layer_call_fn_9236204inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_dense_2_layer_call_and_return_conditional_losses_9236243inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
0
Ы0
Ь1"
trackable_list_wrapper
0
Ы0
Ь1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
ь
trace_02Э
0__inference_spatial_layer1_layer_call_fn_9236594
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02ш
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_9236624
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
Ћ	variables
Ќtrainable_variables
­regularization_losses
Џ__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
ш
Ѓtrace_02Щ
,__inference_activation_layer_call_fn_9236629
В
FullArgSpec
args

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
annotationsЊ *
 zЃtrace_0

Єtrace_02ф
G__inference_activation_layer_call_and_return_conditional_losses_9236642
В
FullArgSpec
args

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
annotationsЊ *
 zЄtrace_0
0
Э0
Ю1"
trackable_list_wrapper
0
Э0
Ю1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ѕnon_trainable_variables
Іlayers
Їmetrics
 Јlayer_regularization_losses
Љlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
ь
Њtrace_02Э
0__inference_spatial_layer2_layer_call_fn_9236651
В
FullArgSpec
args

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
annotationsЊ *
 zЊtrace_0

Ћtrace_02ш
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_9236681
В
FullArgSpec
args

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
annotationsЊ *
 zЋtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ќnon_trainable_variables
­layers
Ўmetrics
 Џlayer_regularization_losses
Аlayer_metrics
З	variables
Иtrainable_variables
Йregularization_losses
Л__call__
+М&call_and_return_all_conditional_losses
'М"call_and_return_conditional_losses"
_generic_user_object
ъ
Бtrace_02Ы
.__inference_activation_1_layer_call_fn_9236686
В
FullArgSpec
args

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
annotationsЊ *
 zБtrace_0

Вtrace_02ц
I__inference_activation_1_layer_call_and_return_conditional_losses_9236699
В
FullArgSpec
args

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
annotationsЊ *
 zВtrace_0
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
B
8__inference_spatial_transformation_layer_call_fn_9234564spatial_layer1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
8__inference_spatial_transformation_layer_call_fn_9234577spatial_layer1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234535spatial_layer1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
B
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234551spatial_layer1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
гBа
)__inference_dense_1_layer_call_fn_9236252inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_dense_1_layer_call_and_return_conditional_losses_9236291inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
гBа
)__inference_dense_3_layer_call_fn_9236300inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_dense_3_layer_call_and_return_conditional_losses_9236339inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
 "
trackable_list_wrapper
K
{0
|1
}2
~3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ЦBУ
6__inference_multi_head_attention_layer_call_fn_9236362queryvaluekey"ѓ
ьВш
FullArgSpecp
argshe
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ЦBУ
6__inference_multi_head_attention_layer_call_fn_9236385queryvaluekey"ѓ
ьВш
FullArgSpecp
argshe
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
сBо
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9236421queryvaluekey"ѓ
ьВш
FullArgSpecp
argshe
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
сBо
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9236457queryvaluekey"ѓ
ьВш
FullArgSpecp
argshe
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

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
Я0
а1"
trackable_list_wrapper
0
Я0
а1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Гnon_trainable_variables
Дlayers
Еmetrics
 Жlayer_regularization_losses
Зlayer_metrics
н	variables
оtrainable_variables
пregularization_losses
с__call__
+т&call_and_return_all_conditional_losses
'т"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
0
б0
в1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Иnon_trainable_variables
Йlayers
Кmetrics
 Лlayer_regularization_losses
Мlayer_metrics
х	variables
цtrainable_variables
чregularization_losses
щ__call__
+ъ&call_and_return_all_conditional_losses
'ъ"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
г0
д1"
trackable_list_wrapper
0
г0
д1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Нnon_trainable_variables
Оlayers
Пmetrics
 Рlayer_regularization_losses
Сlayer_metrics
э	variables
юtrainable_variables
яregularization_losses
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
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
И
Тnon_trainable_variables
Уlayers
Фmetrics
 Хlayer_regularization_losses
Цlayer_metrics
ѕ	variables
іtrainable_variables
їregularization_losses
љ__call__
+њ&call_and_return_all_conditional_losses
'њ"call_and_return_conditional_losses"
_generic_user_object
Ћ2ЈЅ
В
FullArgSpec
args
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Ћ2ЈЅ
В
FullArgSpec
args
jinputs
jmask
varargs
 
varkw
 
defaultsЂ

 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
Чnon_trainable_variables
Шlayers
Щmetrics
 Ъlayer_regularization_losses
Ыlayer_metrics
ћ	variables
ќtrainable_variables
§regularization_losses
џ__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
Џ2ЌЉ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
Џ2ЌЉ
ЂВ
FullArgSpec!
args
jinputs

jtraining
varargs
 
varkw
 
defaultsЂ
p 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
"
_generic_user_object
0
е0
ж1"
trackable_list_wrapper
0
е0
ж1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
Ьnon_trainable_variables
Эlayers
Юmetrics
 Яlayer_regularization_losses
аlayer_metrics
	variables
trainable_variables
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
2
В
FullArgSpec
args

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
annotationsЊ *
 
2
В
FullArgSpec
args

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
annotationsЊ *
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
лBи
%__inference_add_layer_call_fn_9236463inputs_0inputs_1"
В
FullArgSpec
args

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
annotationsЊ *
 
іBѓ
@__inference_add_layer_call_and_return_conditional_losses_9236469inputs_0inputs_1"
В
FullArgSpec
args

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
annotationsЊ *
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
гBа
)__inference_dense_4_layer_call_fn_9236478inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_dense_4_layer_call_and_return_conditional_losses_9236517inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
гBа
)__inference_flatten_layer_call_fn_9236522inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
юBы
D__inference_flatten_layer_call_and_return_conditional_losses_9236528inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
0
з0
и1"
trackable_list_wrapper
0
з0
и1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
	variables
 trainable_variables
Ёregularization_losses
Ѓ__call__
+Є&call_and_return_all_conditional_losses
'Є"call_and_return_conditional_losses"
_generic_user_object
с
жtrace_02Т
%__inference_ou1_layer_call_fn_9236708
В
FullArgSpec
args

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
annotationsЊ *
 zжtrace_0
ќ
зtrace_02н
@__inference_ou1_layer_call_and_return_conditional_losses_9236718
В
FullArgSpec
args

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
annotationsЊ *
 zзtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
иnon_trainable_variables
йlayers
кmetrics
 лlayer_regularization_losses
мlayer_metrics
Ѕ	variables
Іtrainable_variables
Їregularization_losses
Љ__call__
+Њ&call_and_return_all_conditional_losses
'Њ"call_and_return_conditional_losses"
_generic_user_object
ъ
нtrace_02Ы
.__inference_activation_2_layer_call_fn_9236723
В
FullArgSpec
args

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
annotationsЊ *
 zнtrace_0

оtrace_02ц
I__inference_activation_2_layer_call_and_return_conditional_losses_9236736
В
FullArgSpec
args

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
annotationsЊ *
 zоtrace_0
0
й0
к1"
trackable_list_wrapper
0
й0
к1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
пnon_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
Ћ	variables
Ќtrainable_variables
­regularization_losses
Џ__call__
+А&call_and_return_all_conditional_losses
'А"call_and_return_conditional_losses"
_generic_user_object
с
фtrace_02Т
%__inference_ou2_layer_call_fn_9236745
В
FullArgSpec
args

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
annotationsЊ *
 zфtrace_0
ќ
хtrace_02н
@__inference_ou2_layer_call_and_return_conditional_losses_9236755
В
FullArgSpec
args

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
annotationsЊ *
 zхtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
цnon_trainable_variables
чlayers
шmetrics
 щlayer_regularization_losses
ъlayer_metrics
Б	variables
Вtrainable_variables
Гregularization_losses
Е__call__
+Ж&call_and_return_all_conditional_losses
'Ж"call_and_return_conditional_losses"
_generic_user_object
ъ
ыtrace_02Ы
.__inference_activation_3_layer_call_fn_9236760
В
FullArgSpec
args

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
annotationsЊ *
 zыtrace_0

ьtrace_02ц
I__inference_activation_3_layer_call_and_return_conditional_losses_9236773
В
FullArgSpec
args

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
annotationsЊ *
 zьtrace_0
 "
trackable_list_wrapper
@
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
#__inference_U_layer_call_fn_9234696	ou1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
#__inference_U_layer_call_fn_9234709	ou1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
>__inference_U_layer_call_and_return_conditional_losses_9234667	ou1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
>__inference_U_layer_call_and_return_conditional_losses_9234683	ou1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
л0
м1"
trackable_list_wrapper
0
л0
м1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
эnon_trainable_variables
юlayers
яmetrics
 №layer_regularization_losses
ёlayer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
с
ђtrace_02Т
%__inference_ov1_layer_call_fn_9236782
В
FullArgSpec
args

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
annotationsЊ *
 zђtrace_0
ќ
ѓtrace_02н
@__inference_ov1_layer_call_and_return_conditional_losses_9236792
В
FullArgSpec
args

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
annotationsЊ *
 zѓtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
єnon_trainable_variables
ѕlayers
іmetrics
 їlayer_regularization_losses
јlayer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
ъ
љtrace_02Ы
.__inference_activation_4_layer_call_fn_9236797
В
FullArgSpec
args

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
annotationsЊ *
 zљtrace_0

њtrace_02ц
I__inference_activation_4_layer_call_and_return_conditional_losses_9236810
В
FullArgSpec
args

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
annotationsЊ *
 zњtrace_0
0
н0
о1"
trackable_list_wrapper
0
н0
о1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
ћnon_trainable_variables
ќlayers
§metrics
 ўlayer_regularization_losses
џlayer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
с
trace_02Т
%__inference_ov2_layer_call_fn_9236819
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
ќ
trace_02н
@__inference_ov2_layer_call_and_return_conditional_losses_9236829
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_activation_5_layer_call_fn_9236834
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02ц
I__inference_activation_5_layer_call_and_return_conditional_losses_9236847
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
@
0
 1
Ё2
Ђ3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
#__inference_V_layer_call_fn_9234828	ov1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
#__inference_V_layer_call_fn_9234841	ov1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
>__inference_V_layer_call_and_return_conditional_losses_9234799	ov1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
>__inference_V_layer_call_and_return_conditional_losses_9234815	ov1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
0
п0
р1"
trackable_list_wrapper
0
п0
р1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
с
trace_02Т
%__inference_op1_layer_call_fn_9236856
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
ќ
trace_02н
@__inference_op1_layer_call_and_return_conditional_losses_9236866
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
ч	variables
шtrainable_variables
щregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
ъ
trace_02Ы
.__inference_activation_6_layer_call_fn_9236871
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0

trace_02ц
I__inference_activation_6_layer_call_and_return_conditional_losses_9236884
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
0
с0
т1"
trackable_list_wrapper
0
с0
т1"
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
metrics
 layer_regularization_losses
layer_metrics
э	variables
юtrainable_variables
яregularization_losses
ё__call__
+ђ&call_and_return_all_conditional_losses
'ђ"call_and_return_conditional_losses"
_generic_user_object
с
trace_02Т
%__inference_op2_layer_call_fn_9236893
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
ќ
trace_02н
@__inference_op2_layer_call_and_return_conditional_losses_9236903
В
FullArgSpec
args

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
annotationsЊ *
 ztrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
И
non_trainable_variables
layers
 metrics
 Ёlayer_regularization_losses
Ђlayer_metrics
ѓ	variables
єtrainable_variables
ѕregularization_losses
ї__call__
+ј&call_and_return_all_conditional_losses
'ј"call_and_return_conditional_losses"
_generic_user_object
ъ
Ѓtrace_02Ы
.__inference_activation_7_layer_call_fn_9236908
В
FullArgSpec
args

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
annotationsЊ *
 zЃtrace_0

Єtrace_02ц
I__inference_activation_7_layer_call_and_return_conditional_losses_9236921
В
FullArgSpec
args

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
annotationsЊ *
 zЄtrace_0
 "
trackable_list_wrapper
@
Љ0
Њ1
Ћ2
Ќ3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
#__inference_P_layer_call_fn_9234960	op1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
фBс
#__inference_P_layer_call_fn_9234973	op1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
>__inference_P_layer_call_and_return_conditional_losses_9234931	op1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
џBќ
>__inference_P_layer_call_and_return_conditional_losses_9234947	op1_input"Ќ
ЅВЁ
FullArgSpec)
args!
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
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
дBб
*__inference_output_u_layer_call_fn_9236537inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
яBь
E__inference_output_u_layer_call_and_return_conditional_losses_9236547inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
дBб
*__inference_output_v_layer_call_fn_9236556inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
яBь
E__inference_output_v_layer_call_and_return_conditional_losses_9236566inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
дBб
*__inference_output_p_layer_call_fn_9236575inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
яBь
E__inference_output_p_layer_call_and_return_conditional_losses_9236585inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
кBз
0__inference_spatial_layer1_layer_call_fn_9236594inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѕBђ
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_9236624inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
жBг
,__inference_activation_layer_call_fn_9236629inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ёBю
G__inference_activation_layer_call_and_return_conditional_losses_9236642inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
кBз
0__inference_spatial_layer2_layer_call_fn_9236651inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѕBђ
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_9236681inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
иBе
.__inference_activation_1_layer_call_fn_9236686inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѓB№
I__inference_activation_1_layer_call_and_return_conditional_losses_9236699inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
ЯBЬ
%__inference_ou1_layer_call_fn_9236708inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ъBч
@__inference_ou1_layer_call_and_return_conditional_losses_9236718inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
иBе
.__inference_activation_2_layer_call_fn_9236723inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѓB№
I__inference_activation_2_layer_call_and_return_conditional_losses_9236736inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
ЯBЬ
%__inference_ou2_layer_call_fn_9236745inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ъBч
@__inference_ou2_layer_call_and_return_conditional_losses_9236755inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
иBе
.__inference_activation_3_layer_call_fn_9236760inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѓB№
I__inference_activation_3_layer_call_and_return_conditional_losses_9236773inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
ЯBЬ
%__inference_ov1_layer_call_fn_9236782inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ъBч
@__inference_ov1_layer_call_and_return_conditional_losses_9236792inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
иBе
.__inference_activation_4_layer_call_fn_9236797inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѓB№
I__inference_activation_4_layer_call_and_return_conditional_losses_9236810inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
ЯBЬ
%__inference_ov2_layer_call_fn_9236819inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ъBч
@__inference_ov2_layer_call_and_return_conditional_losses_9236829inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
иBе
.__inference_activation_5_layer_call_fn_9236834inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѓB№
I__inference_activation_5_layer_call_and_return_conditional_losses_9236847inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
ЯBЬ
%__inference_op1_layer_call_fn_9236856inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ъBч
@__inference_op1_layer_call_and_return_conditional_losses_9236866inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
иBе
.__inference_activation_6_layer_call_fn_9236871inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѓB№
I__inference_activation_6_layer_call_and_return_conditional_losses_9236884inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
ЯBЬ
%__inference_op2_layer_call_fn_9236893inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ъBч
@__inference_op2_layer_call_and_return_conditional_losses_9236903inputs"
В
FullArgSpec
args

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
annotationsЊ *
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
иBе
.__inference_activation_7_layer_call_fn_9236908inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
ѓB№
I__inference_activation_7_layer_call_and_return_conditional_losses_9236921inputs"
В
FullArgSpec
args

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
annotationsЊ *
 
UbS
beta:0I__inference_activation_7_layer_call_and_return_conditional_losses_9236921
WbU
inputs:0I__inference_activation_7_layer_call_and_return_conditional_losses_9236921
UbS
beta:0I__inference_activation_7_layer_call_and_return_conditional_losses_9234928
WbU
inputs:0I__inference_activation_7_layer_call_and_return_conditional_losses_9234928
UbS
beta:0I__inference_activation_6_layer_call_and_return_conditional_losses_9236884
WbU
inputs:0I__inference_activation_6_layer_call_and_return_conditional_losses_9236884
UbS
beta:0I__inference_activation_6_layer_call_and_return_conditional_losses_9234899
WbU
inputs:0I__inference_activation_6_layer_call_and_return_conditional_losses_9234899
UbS
beta:0I__inference_activation_5_layer_call_and_return_conditional_losses_9236847
WbU
inputs:0I__inference_activation_5_layer_call_and_return_conditional_losses_9236847
UbS
beta:0I__inference_activation_5_layer_call_and_return_conditional_losses_9234796
WbU
inputs:0I__inference_activation_5_layer_call_and_return_conditional_losses_9234796
UbS
beta:0I__inference_activation_4_layer_call_and_return_conditional_losses_9236810
WbU
inputs:0I__inference_activation_4_layer_call_and_return_conditional_losses_9236810
UbS
beta:0I__inference_activation_4_layer_call_and_return_conditional_losses_9234767
WbU
inputs:0I__inference_activation_4_layer_call_and_return_conditional_losses_9234767
UbS
beta:0I__inference_activation_3_layer_call_and_return_conditional_losses_9236773
WbU
inputs:0I__inference_activation_3_layer_call_and_return_conditional_losses_9236773
UbS
beta:0I__inference_activation_3_layer_call_and_return_conditional_losses_9234664
WbU
inputs:0I__inference_activation_3_layer_call_and_return_conditional_losses_9234664
UbS
beta:0I__inference_activation_2_layer_call_and_return_conditional_losses_9236736
WbU
inputs:0I__inference_activation_2_layer_call_and_return_conditional_losses_9236736
UbS
beta:0I__inference_activation_2_layer_call_and_return_conditional_losses_9234635
WbU
inputs:0I__inference_activation_2_layer_call_and_return_conditional_losses_9234635
UbS
beta:0I__inference_activation_1_layer_call_and_return_conditional_losses_9236699
WbU
inputs:0I__inference_activation_1_layer_call_and_return_conditional_losses_9236699
UbS
beta:0I__inference_activation_1_layer_call_and_return_conditional_losses_9234532
WbU
inputs:0I__inference_activation_1_layer_call_and_return_conditional_losses_9234532
SbQ
beta:0G__inference_activation_layer_call_and_return_conditional_losses_9236642
UbS
inputs:0G__inference_activation_layer_call_and_return_conditional_losses_9236642
SbQ
beta:0G__inference_activation_layer_call_and_return_conditional_losses_9234483
UbS
inputs:0G__inference_activation_layer_call_and_return_conditional_losses_9234483
PbN
beta:0D__inference_dense_4_layer_call_and_return_conditional_losses_9236517
SbQ
	BiasAdd:0D__inference_dense_4_layer_call_and_return_conditional_losses_9236517
PbN
beta:0D__inference_dense_4_layer_call_and_return_conditional_losses_9235358
SbQ
	BiasAdd:0D__inference_dense_4_layer_call_and_return_conditional_losses_9235358
PbN
beta:0D__inference_dense_3_layer_call_and_return_conditional_losses_9236339
SbQ
	BiasAdd:0D__inference_dense_3_layer_call_and_return_conditional_losses_9236339
PbN
beta:0D__inference_dense_3_layer_call_and_return_conditional_losses_9235254
SbQ
	BiasAdd:0D__inference_dense_3_layer_call_and_return_conditional_losses_9235254
PbN
beta:0D__inference_dense_1_layer_call_and_return_conditional_losses_9236291
SbQ
	BiasAdd:0D__inference_dense_1_layer_call_and_return_conditional_losses_9236291
PbN
beta:0D__inference_dense_1_layer_call_and_return_conditional_losses_9235210
SbQ
	BiasAdd:0D__inference_dense_1_layer_call_and_return_conditional_losses_9235210
PbN
beta:0D__inference_dense_2_layer_call_and_return_conditional_losses_9236243
SbQ
	BiasAdd:0D__inference_dense_2_layer_call_and_return_conditional_losses_9236243
PbN
beta:0D__inference_dense_2_layer_call_and_return_conditional_losses_9235099
SbQ
	BiasAdd:0D__inference_dense_2_layer_call_and_return_conditional_losses_9235099
NbL
beta:0B__inference_dense_layer_call_and_return_conditional_losses_9236195
QbO
	BiasAdd:0B__inference_dense_layer_call_and_return_conditional_losses_9236195
NbL
beta:0B__inference_dense_layer_call_and_return_conditional_losses_9235143
QbO
	BiasAdd:0B__inference_dense_layer_call_and_return_conditional_losses_9235143
<b:
model/dense_2/beta:0"__inference__wrapped_model_9234433
?b=
model/dense_2/BiasAdd:0"__inference__wrapped_model_9234433
:b8
model/dense/beta:0"__inference__wrapped_model_9234433
=b;
model/dense/BiasAdd:0"__inference__wrapped_model_9234433
VbT
.model/spatial_transformation/activation/beta:0"__inference__wrapped_model_9234433
]b[
5model/spatial_transformation/spatial_layer1/BiasAdd:0"__inference__wrapped_model_9234433
XbV
0model/spatial_transformation/activation_1/beta:0"__inference__wrapped_model_9234433
]b[
5model/spatial_transformation/spatial_layer2/BiasAdd:0"__inference__wrapped_model_9234433
<b:
model/dense_1/beta:0"__inference__wrapped_model_9234433
?b=
model/dense_1/BiasAdd:0"__inference__wrapped_model_9234433
<b:
model/dense_3/beta:0"__inference__wrapped_model_9234433
?b=
model/dense_3/BiasAdd:0"__inference__wrapped_model_9234433
<b:
model/dense_4/beta:0"__inference__wrapped_model_9234433
?b=
model/dense_4/BiasAdd:0"__inference__wrapped_model_9234433
CbA
model/P/activation_6/beta:0"__inference__wrapped_model_9234433
=b;
model/P/op1/BiasAdd:0"__inference__wrapped_model_9234433
CbA
model/P/activation_7/beta:0"__inference__wrapped_model_9234433
=b;
model/P/op2/BiasAdd:0"__inference__wrapped_model_9234433
CbA
model/V/activation_4/beta:0"__inference__wrapped_model_9234433
=b;
model/V/ov1/BiasAdd:0"__inference__wrapped_model_9234433
CbA
model/V/activation_5/beta:0"__inference__wrapped_model_9234433
=b;
model/V/ov2/BiasAdd:0"__inference__wrapped_model_9234433
CbA
model/U/activation_2/beta:0"__inference__wrapped_model_9234433
=b;
model/U/ou1/BiasAdd:0"__inference__wrapped_model_9234433
CbA
model/U/activation_3/beta:0"__inference__wrapped_model_9234433
=b;
model/U/ou2/BiasAdd:0"__inference__wrapped_model_9234433Ж
>__inference_P_layer_call_and_return_conditional_losses_9234931tпрст:Ђ7
0Ђ-
# 
	op1_inputџџџџџџџџџ@
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 Ж
>__inference_P_layer_call_and_return_conditional_losses_9234947tпрст:Ђ7
0Ђ-
# 
	op1_inputџџџџџџџџџ@
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
#__inference_P_layer_call_fn_9234960iпрст:Ђ7
0Ђ-
# 
	op1_inputџџџџџџџџџ@
p

 
Њ "!
unknownџџџџџџџџџ@
#__inference_P_layer_call_fn_9234973iпрст:Ђ7
0Ђ-
# 
	op1_inputџџџџџџџџџ@
p 

 
Њ "!
unknownџџџџџџџџџ@Ж
>__inference_U_layer_call_and_return_conditional_losses_9234667tзийк:Ђ7
0Ђ-
# 
	ou1_inputџџџџџџџџџ@
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 Ж
>__inference_U_layer_call_and_return_conditional_losses_9234683tзийк:Ђ7
0Ђ-
# 
	ou1_inputџџџџџџџџџ@
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
#__inference_U_layer_call_fn_9234696iзийк:Ђ7
0Ђ-
# 
	ou1_inputџџџџџџџџџ@
p

 
Њ "!
unknownџџџџџџџџџ@
#__inference_U_layer_call_fn_9234709iзийк:Ђ7
0Ђ-
# 
	ou1_inputџџџџџџџџџ@
p 

 
Њ "!
unknownџџџџџџџџџ@Ж
>__inference_V_layer_call_and_return_conditional_losses_9234799tлмно:Ђ7
0Ђ-
# 
	ov1_inputџџџџџџџџџ@
p

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 Ж
>__inference_V_layer_call_and_return_conditional_losses_9234815tлмно:Ђ7
0Ђ-
# 
	ov1_inputџџџџџџџџџ@
p 

 
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
#__inference_V_layer_call_fn_9234828iлмно:Ђ7
0Ђ-
# 
	ov1_inputџџџџџџџџџ@
p

 
Њ "!
unknownџџџџџџџџџ@
#__inference_V_layer_call_fn_9234841iлмно:Ђ7
0Ђ-
# 
	ov1_inputџџџџџџџџџ@
p 

 
Њ "!
unknownџџџџџџџџџ@З
"__inference__wrapped_model_9234433HYZQRЫЬЭЮklstЯабвгдежпрстлмнозийкЩЪСТЙКЌЂЈ
 Ђ

!
X_layerџџџџџџџџџ
!
Y_layerџџџџџџџџџ
!
T_layerџџџџџџџџџ
0-
	Xbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ybc_layerџџџџџџџџџџџџџџџџџџ
0-
	Tbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ubc_layerџџџџџџџџџџџџџџџџџџ
0-
	Vbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Pbc_layerџџџџџџџџџџџџџџџџџџ
Њ "Њ
.
output_p"
output_pџџџџџџџџџ
.
output_u"
output_uџџџџџџџџџ
.
output_v"
output_vџџџџџџџџџД
I__inference_activation_1_layer_call_and_return_conditional_losses_9236699g3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 
.__inference_activation_1_layer_call_fn_9236686\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "%"
unknownџџџџџџџџџ@Ќ
I__inference_activation_2_layer_call_and_return_conditional_losses_9236736_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
.__inference_activation_2_layer_call_fn_9236723T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Ќ
I__inference_activation_3_layer_call_and_return_conditional_losses_9236773_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
.__inference_activation_3_layer_call_fn_9236760T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Ќ
I__inference_activation_4_layer_call_and_return_conditional_losses_9236810_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
.__inference_activation_4_layer_call_fn_9236797T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Ќ
I__inference_activation_5_layer_call_and_return_conditional_losses_9236847_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
.__inference_activation_5_layer_call_fn_9236834T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Ќ
I__inference_activation_6_layer_call_and_return_conditional_losses_9236884_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
.__inference_activation_6_layer_call_fn_9236871T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Ќ
I__inference_activation_7_layer_call_and_return_conditional_losses_9236921_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
.__inference_activation_7_layer_call_fn_9236908T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@В
G__inference_activation_layer_call_and_return_conditional_losses_9236642g3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 
,__inference_activation_layer_call_fn_9236629\3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "%"
unknownџџџџџџџџџ@л
@__inference_add_layer_call_and_return_conditional_losses_9236469bЂ_
XЂU
SP
&#
inputs_0џџџџџџџџџ@
&#
inputs_1џџџџџџџџџ@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 Е
%__inference_add_layer_call_fn_9236463bЂ_
XЂU
SP
&#
inputs_0џџџџџџџџџ@
&#
inputs_1џџџџџџџџџ@
Њ "%"
unknownџџџџџџџџџ@З
J__inference_concatenate_1_layer_call_and_return_conditional_losses_9236114шЊЂІ
Ђ

/,
inputs_0џџџџџџџџџџџџџџџџџџ
/,
inputs_1џџџџџџџџџџџџџџџџџџ
/,
inputs_2џџџџџџџџџџџџџџџџџџ
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ
 
/__inference_concatenate_1_layer_call_fn_9236106нЊЂІ
Ђ

/,
inputs_0џџџџџџџџџџџџџџџџџџ
/,
inputs_1џџџџџџџџџџџџџџџџџџ
/,
inputs_2џџџџџџџџџџџџџџџџџџ
Њ ".+
unknownџџџџџџџџџџџџџџџџџџЗ
J__inference_concatenate_2_layer_call_and_return_conditional_losses_9236129шЊЂІ
Ђ

/,
inputs_0џџџџџџџџџџџџџџџџџџ
/,
inputs_1џџџџџџџџџџџџџџџџџџ
/,
inputs_2џџџџџџџџџџџџџџџџџџ
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ
 
/__inference_concatenate_2_layer_call_fn_9236121нЊЂІ
Ђ

/,
inputs_0џџџџџџџџџџџџџџџџџџ
/,
inputs_1џџџџџџџџџџџџџџџџџџ
/,
inputs_2џџџџџџџџџџџџџџџџџџ
Њ ".+
unknownџџџџџџџџџџџџџџџџџџћ
H__inference_concatenate_layer_call_and_return_conditional_losses_9236099Ў~Ђ{
tЂq
ol
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 е
-__inference_concatenate_layer_call_fn_9236091Ѓ~Ђ{
tЂq
ol
"
inputs_0џџџџџџџџџ
"
inputs_1џџџџџџџџџ
"
inputs_2џџџџџџџџџ
Њ "!
unknownџџџџџџџџџХ
D__inference_dense_1_layer_call_and_return_conditional_losses_9236291}kl<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ@
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 
)__inference_dense_1_layer_call_fn_9236252rkl<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ@
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@Х
D__inference_dense_2_layer_call_and_return_conditional_losses_9236243}YZ<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 
)__inference_dense_2_layer_call_fn_9236204rYZ<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@Х
D__inference_dense_3_layer_call_and_return_conditional_losses_9236339}st<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ@
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 
)__inference_dense_3_layer_call_fn_9236300rst<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ@
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@Е
D__inference_dense_4_layer_call_and_return_conditional_losses_9236517m3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 
)__inference_dense_4_layer_call_fn_9236478b3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "%"
unknownџџџџџџџџџ@У
B__inference_dense_layer_call_and_return_conditional_losses_9236195}QR<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ@
 
'__inference_dense_layer_call_fn_9236156rQR<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ
Њ ".+
unknownџџџџџџџџџџџџџџџџџџ@Ћ
D__inference_flatten_layer_call_and_return_conditional_losses_9236528c3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
)__inference_flatten_layer_call_fn_9236522X3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@я
$__inference_internal_grad_fn_9237041ЦЅІ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237068ЦЇЈ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237095ЦЉЊ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237122ЦЋЌ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237149Ц­Ў~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237176ЦЏА~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237203ЦБВ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237230ЦГД~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237257ЦЕЖ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237284ЦЗИ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237311ЦЙК~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237338ЦЛМ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237365дНОЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237392дПРЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237419дСТЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237446дУФЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237473дХЦЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237500дЧШЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237527ёЩЪЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237554ёЫЬЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237581ёЭЮЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237608ёЯаЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237635ёбвЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237662ёгдЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237689ёежЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237716ёзиЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237743ёйкЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237770ёлмЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237797дноЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237824дпрЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237851ёстЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 
$__inference_internal_grad_fn_9237878ёуфЂ
Ђ

 
52
result_grads_0џџџџџџџџџџџџџџџџџџ@
52
result_grads_1џџџџџџџџџџџџџџџџџџ@

result_grads_2 
Њ "KH

 
/,
tensor_1џџџџџџџџџџџџџџџџџџ@

tensor_2 §
$__inference_internal_grad_fn_9237905дхцЂ
|Ђy

 
,)
result_grads_0џџџџџџџџџ@
,)
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ "B?

 
&#
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237932Цчш~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237959Цщъ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9237986Цыь~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9238013Цэю~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9238040Ця№~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 я
$__inference_internal_grad_fn_9238067Цёђ~Ђ{
tЂq

 
(%
result_grads_0џџџџџџџџџ@
(%
result_grads_1џџџџџџџџџ@

result_grads_2 
Њ ">;

 
"
tensor_1џџџџџџџџџ@

tensor_2 Щ
B__inference_model_layer_call_and_return_conditional_losses_9235446HYZQRЫЬЭЮklstЯабвгдежпрстлмнозийкЩЪСТЙКДЂА
ЈЂЄ

!
X_layerџџџџџџџџџ
!
Y_layerџџџџџџџџџ
!
T_layerџџџџџџџџџ
0-
	Xbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ybc_layerџџџџџџџџџџџџџџџџџџ
0-
	Tbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ubc_layerџџџџџџџџџџџџџџџџџџ
0-
	Vbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Pbc_layerџџџџџџџџџџџџџџџџџџ
p

 
Њ "Ђ|
ur
$!

tensor_0_0џџџџџџџџџ
$!

tensor_0_1џџџџџџџџџ
$!

tensor_0_2џџџџџџџџџ
 Щ
B__inference_model_layer_call_and_return_conditional_losses_9235597HYZQRЫЬЭЮklstЯабвгдежпрстлмнозийкЩЪСТЙКДЂА
ЈЂЄ

!
X_layerџџџџџџџџџ
!
Y_layerџџџџџџџџџ
!
T_layerџџџџџџџџџ
0-
	Xbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ybc_layerџџџџџџџџџџџџџџџџџџ
0-
	Tbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ubc_layerџџџџџџџџџџџџџџџџџџ
0-
	Vbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Pbc_layerџџџџџџџџџџџџџџџџџџ
p 

 
Њ "Ђ|
ur
$!

tensor_0_0џџџџџџџџџ
$!

tensor_0_1џџџџџџџџџ
$!

tensor_0_2џџџџџџџџџ
 
'__inference_model_layer_call_fn_9235694ђHYZQRЫЬЭЮklstЯабвгдежпрстлмнозийкЩЪСТЙКДЂА
ЈЂЄ

!
X_layerџџџџџџџџџ
!
Y_layerџџџџџџџџџ
!
T_layerџџџџџџџџџ
0-
	Xbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ybc_layerџџџџџџџџџџџџџџџџџџ
0-
	Tbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ubc_layerџџџџџџџџџџџџџџџџџџ
0-
	Vbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Pbc_layerџџџџџџџџџџџџџџџџџџ
p

 
Њ "ol
"
tensor_0џџџџџџџџџ
"
tensor_1џџџџџџџџџ
"
tensor_2џџџџџџџџџ
'__inference_model_layer_call_fn_9235791ђHYZQRЫЬЭЮklstЯабвгдежпрстлмнозийкЩЪСТЙКДЂА
ЈЂЄ

!
X_layerџџџџџџџџџ
!
Y_layerџџџџџџџџџ
!
T_layerџџџџџџџџџ
0-
	Xbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ybc_layerџџџџџџџџџџџџџџџџџџ
0-
	Tbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Ubc_layerџџџџџџџџџџџџџџџџџџ
0-
	Vbc_layerџџџџџџџџџџџџџџџџџџ
0-
	Pbc_layerџџџџџџџџџџџџџџџџџџ
p 

 
Њ "ol
"
tensor_0џџџџџџџџџ
"
tensor_1џџџџџџџџџ
"
tensor_2џџџџџџџџџМ
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9236421цЯабвгдежЂ
Ђ
# 
queryџџџџџџџџџ@
,)
valueџџџџџџџџџџџџџџџџџџ@
*'
keyџџџџџџџџџџџџџџџџџџ@

 
p 
p
p 
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 М
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_9236457цЯабвгдежЂ
Ђ
# 
queryџџџџџџџџџ@
,)
valueџџџџџџџџџџџџџџџџџџ@
*'
keyџџџџџџџџџџџџџџџџџџ@

 
p 
p 
p 
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 
6__inference_multi_head_attention_layer_call_fn_9236362лЯабвгдежЂ
Ђ
# 
queryџџџџџџџџџ@
,)
valueџџџџџџџџџџџџџџџџџџ@
*'
keyџџџџџџџџџџџџџџџџџџ@

 
p 
p
p 
Њ "%"
unknownџџџџџџџџџ@
6__inference_multi_head_attention_layer_call_fn_9236385лЯабвгдежЂ
Ђ
# 
queryџџџџџџџџџ@
,)
valueџџџџџџџџџџџџџџџџџџ@
*'
keyџџџџџџџџџџџџџџџџџџ@

 
p 
p 
p 
Њ "%"
unknownџџџџџџџџџ@Љ
@__inference_op1_layer_call_and_return_conditional_losses_9236866eпр/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
%__inference_op1_layer_call_fn_9236856Zпр/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Љ
@__inference_op2_layer_call_and_return_conditional_losses_9236903eст/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
%__inference_op2_layer_call_fn_9236893Zст/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Љ
@__inference_ou1_layer_call_and_return_conditional_losses_9236718eзи/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
%__inference_ou1_layer_call_fn_9236708Zзи/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Љ
@__inference_ou2_layer_call_and_return_conditional_losses_9236755eйк/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
%__inference_ou2_layer_call_fn_9236745Zйк/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Ў
E__inference_output_p_layer_call_and_return_conditional_losses_9236585eЩЪ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
*__inference_output_p_layer_call_fn_9236575ZЩЪ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЎ
E__inference_output_u_layer_call_and_return_conditional_losses_9236547eЙК/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
*__inference_output_u_layer_call_fn_9236537ZЙК/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЎ
E__inference_output_v_layer_call_and_return_conditional_losses_9236566eСТ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
*__inference_output_v_layer_call_fn_9236556ZСТ/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџЉ
@__inference_ov1_layer_call_and_return_conditional_losses_9236792eлм/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
%__inference_ov1_layer_call_fn_9236782Zлм/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Љ
@__inference_ov2_layer_call_and_return_conditional_losses_9236829eно/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ ",Ђ)
"
tensor_0џџџџџџџџџ@
 
%__inference_ov2_layer_call_fn_9236819Zно/Ђ,
%Ђ"
 
inputsџџџџџџџџџ@
Њ "!
unknownџџџџџџџџџ@Х
H__inference_rescaling_1_layer_call_and_return_conditional_losses_9236084y<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ
Њ "9Ђ6
/,
tensor_0џџџџџџџџџџџџџџџџџџ
 
-__inference_rescaling_1_layer_call_fn_9236074n<Ђ9
2Ђ/
-*
inputsџџџџџџџџџџџџџџџџџџ
Њ ".+
unknownџџџџџџџџџџџџџџџџџџЉ
F__inference_rescaling_layer_call_and_return_conditional_losses_9236069_/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ ",Ђ)
"
tensor_0џџџџџџџџџ
 
+__inference_rescaling_layer_call_fn_9236059T/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "!
unknownџџџџџџџџџЋ
D__inference_reshape_layer_call_and_return_conditional_losses_9236147c/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ
 
)__inference_reshape_layer_call_fn_9236134X/Ђ,
%Ђ"
 
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџЂ
%__inference_signature_wrapper_9236054јHYZQRЫЬЭЮklstЯабвгдежпрстлмнозийкЩЪСТЙКЂ
Ђ 
Њ
=
	Pbc_layer0-
	pbc_layerџџџџџџџџџџџџџџџџџџ
,
T_layer!
t_layerџџџџџџџџџ
=
	Tbc_layer0-
	tbc_layerџџџџџџџџџџџџџџџџџџ
=
	Ubc_layer0-
	ubc_layerџџџџџџџџџџџџџџџџџџ
=
	Vbc_layer0-
	vbc_layerџџџџџџџџџџџџџџџџџџ
,
X_layer!
x_layerџџџџџџџџџ
=
	Xbc_layer0-
	xbc_layerџџџџџџџџџџџџџџџџџџ
,
Y_layer!
y_layerџџџџџџџџџ
=
	Ybc_layer0-
	ybc_layerџџџџџџџџџџџџџџџџџџ"Њ
.
output_p"
output_pџџџџџџџџџ
.
output_u"
output_uџџџџџџџџџ
.
output_v"
output_vџџџџџџџџџМ
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_9236624mЫЬ3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 
0__inference_spatial_layer1_layer_call_fn_9236594bЫЬ3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ
Њ "%"
unknownџџџџџџџџџ@М
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_9236681mЭЮ3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 
0__inference_spatial_layer2_layer_call_fn_9236651bЭЮ3Ђ0
)Ђ&
$!
inputsџџџџџџџџџ@
Њ "%"
unknownџџџџџџџџџ@п
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234535ЫЬЭЮIЂF
?Ђ<
2/
spatial_layer1_inputџџџџџџџџџ
p

 
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 п
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_9234551ЫЬЭЮIЂF
?Ђ<
2/
spatial_layer1_inputџџџџџџџџџ
p 

 
Њ "0Ђ-
&#
tensor_0џџџџџџџџџ@
 И
8__inference_spatial_transformation_layer_call_fn_9234564|ЫЬЭЮIЂF
?Ђ<
2/
spatial_layer1_inputџџџџџџџџџ
p

 
Њ "%"
unknownџџџџџџџџџ@И
8__inference_spatial_transformation_layer_call_fn_9234577|ЫЬЭЮIЂF
?Ђ<
2/
spatial_layer1_inputџџџџџџџџџ
p 

 
Њ "%"
unknownџџџџџџџџџ@