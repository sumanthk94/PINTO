¤Я)
я├
D
AddV2
x"T
y"T
z"T"
Ttype:
2	АР
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( И
А
BiasAdd

value"T	
bias"T
output"T""
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
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
resourceИ
R
Einsum
inputs"T*N
output"T"
equationstring"
Nint(0"	
Ttype
о
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
Ж
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool("
allow_missing_filesbool( И
?
Mul
x"T
y"T
z"T"
Ttype:
2	Р
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
П
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
dtypetypeИ
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
list(type)(0И
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0И
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
output"out_typeКэout_type"	
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
┴
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
executor_typestring Ии
@
StaticRegexFullMatch	
input

output
"
patternstring
ў
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
░
VarHandleOp
resource"
	containerstring "
shared_namestring "

debug_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 И"serve*2.15.02v2.15.0-2-g0b15fdfcb3f8ёУ!
Г
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
Н

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
Г
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
Н

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
я
,multi_head_attention_2/attention_output/biasVarHandleOp*
_output_shapes
: *=

debug_name/-multi_head_attention_2/attention_output/bias/*
dtype0*
shape:@*=
shared_name.,multi_head_attention_2/attention_output/bias
й
@multi_head_attention_2/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_2/attention_output/bias*
_output_shapes
:@*
dtype0
¤
.multi_head_attention_2/attention_output/kernelVarHandleOp*
_output_shapes
: *?

debug_name1/multi_head_attention_2/attention_output/kernel/*
dtype0*
shape:@@*?
shared_name0.multi_head_attention_2/attention_output/kernel
╡
Bmulti_head_attention_2/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_2/attention_output/kernel*"
_output_shapes
:@@*
dtype0
╥
!multi_head_attention_2/value/biasVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention_2/value/bias/*
dtype0*
shape
:@*2
shared_name#!multi_head_attention_2/value/bias
Ч
5multi_head_attention_2/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/value/bias*
_output_shapes

:@*
dtype0
▄
#multi_head_attention_2/value/kernelVarHandleOp*
_output_shapes
: *4

debug_name&$multi_head_attention_2/value/kernel/*
dtype0*
shape:@@*4
shared_name%#multi_head_attention_2/value/kernel
Я
7multi_head_attention_2/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_2/value/kernel*"
_output_shapes
:@@*
dtype0
╠
multi_head_attention_2/key/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention_2/key/bias/*
dtype0*
shape
:@*0
shared_name!multi_head_attention_2/key/bias
У
3multi_head_attention_2/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_2/key/bias*
_output_shapes

:@*
dtype0
╓
!multi_head_attention_2/key/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention_2/key/kernel/*
dtype0*
shape:@@*2
shared_name#!multi_head_attention_2/key/kernel
Ы
5multi_head_attention_2/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/key/kernel*"
_output_shapes
:@@*
dtype0
╥
!multi_head_attention_2/query/biasVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention_2/query/bias/*
dtype0*
shape
:@*2
shared_name#!multi_head_attention_2/query/bias
Ч
5multi_head_attention_2/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_2/query/bias*
_output_shapes

:@*
dtype0
▄
#multi_head_attention_2/query/kernelVarHandleOp*
_output_shapes
: *4

debug_name&$multi_head_attention_2/query/kernel/*
dtype0*
shape:@@*4
shared_name%#multi_head_attention_2/query/kernel
Я
7multi_head_attention_2/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_2/query/kernel*"
_output_shapes
:@@*
dtype0
я
,multi_head_attention_1/attention_output/biasVarHandleOp*
_output_shapes
: *=

debug_name/-multi_head_attention_1/attention_output/bias/*
dtype0*
shape:@*=
shared_name.,multi_head_attention_1/attention_output/bias
й
@multi_head_attention_1/attention_output/bias/Read/ReadVariableOpReadVariableOp,multi_head_attention_1/attention_output/bias*
_output_shapes
:@*
dtype0
¤
.multi_head_attention_1/attention_output/kernelVarHandleOp*
_output_shapes
: *?

debug_name1/multi_head_attention_1/attention_output/kernel/*
dtype0*
shape:@@*?
shared_name0.multi_head_attention_1/attention_output/kernel
╡
Bmulti_head_attention_1/attention_output/kernel/Read/ReadVariableOpReadVariableOp.multi_head_attention_1/attention_output/kernel*"
_output_shapes
:@@*
dtype0
╥
!multi_head_attention_1/value/biasVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention_1/value/bias/*
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/value/bias
Ч
5multi_head_attention_1/value/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/value/bias*
_output_shapes

:@*
dtype0
▄
#multi_head_attention_1/value/kernelVarHandleOp*
_output_shapes
: *4

debug_name&$multi_head_attention_1/value/kernel/*
dtype0*
shape:@@*4
shared_name%#multi_head_attention_1/value/kernel
Я
7multi_head_attention_1/value/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/value/kernel*"
_output_shapes
:@@*
dtype0
╠
multi_head_attention_1/key/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention_1/key/bias/*
dtype0*
shape
:@*0
shared_name!multi_head_attention_1/key/bias
У
3multi_head_attention_1/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention_1/key/bias*
_output_shapes

:@*
dtype0
╓
!multi_head_attention_1/key/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention_1/key/kernel/*
dtype0*
shape:@@*2
shared_name#!multi_head_attention_1/key/kernel
Ы
5multi_head_attention_1/key/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/key/kernel*"
_output_shapes
:@@*
dtype0
╥
!multi_head_attention_1/query/biasVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention_1/query/bias/*
dtype0*
shape
:@*2
shared_name#!multi_head_attention_1/query/bias
Ч
5multi_head_attention_1/query/bias/Read/ReadVariableOpReadVariableOp!multi_head_attention_1/query/bias*
_output_shapes

:@*
dtype0
▄
#multi_head_attention_1/query/kernelVarHandleOp*
_output_shapes
: *4

debug_name&$multi_head_attention_1/query/kernel/*
dtype0*
shape:@@*4
shared_name%#multi_head_attention_1/query/kernel
Я
7multi_head_attention_1/query/kernel/Read/ReadVariableOpReadVariableOp#multi_head_attention_1/query/kernel*"
_output_shapes
:@@*
dtype0
щ
*multi_head_attention/attention_output/biasVarHandleOp*
_output_shapes
: *;

debug_name-+multi_head_attention/attention_output/bias/*
dtype0*
shape:@*;
shared_name,*multi_head_attention/attention_output/bias
е
>multi_head_attention/attention_output/bias/Read/ReadVariableOpReadVariableOp*multi_head_attention/attention_output/bias*
_output_shapes
:@*
dtype0
ў
,multi_head_attention/attention_output/kernelVarHandleOp*
_output_shapes
: *=

debug_name/-multi_head_attention/attention_output/kernel/*
dtype0*
shape:@@*=
shared_name.,multi_head_attention/attention_output/kernel
▒
@multi_head_attention/attention_output/kernel/Read/ReadVariableOpReadVariableOp,multi_head_attention/attention_output/kernel*"
_output_shapes
:@@*
dtype0
╠
multi_head_attention/value/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/value/bias/*
dtype0*
shape
:@*0
shared_name!multi_head_attention/value/bias
У
3multi_head_attention/value/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/value/bias*
_output_shapes

:@*
dtype0
╓
!multi_head_attention/value/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/value/kernel/*
dtype0*
shape:@@*2
shared_name#!multi_head_attention/value/kernel
Ы
5multi_head_attention/value/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/value/kernel*"
_output_shapes
:@@*
dtype0
╞
multi_head_attention/key/biasVarHandleOp*
_output_shapes
: *.

debug_name multi_head_attention/key/bias/*
dtype0*
shape
:@*.
shared_namemulti_head_attention/key/bias
П
1multi_head_attention/key/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/bias*
_output_shapes

:@*
dtype0
╨
multi_head_attention/key/kernelVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/key/kernel/*
dtype0*
shape:@@*0
shared_name!multi_head_attention/key/kernel
Ч
3multi_head_attention/key/kernel/Read/ReadVariableOpReadVariableOpmulti_head_attention/key/kernel*"
_output_shapes
:@@*
dtype0
╠
multi_head_attention/query/biasVarHandleOp*
_output_shapes
: *0

debug_name" multi_head_attention/query/bias/*
dtype0*
shape
:@*0
shared_name!multi_head_attention/query/bias
У
3multi_head_attention/query/bias/Read/ReadVariableOpReadVariableOpmulti_head_attention/query/bias*
_output_shapes

:@*
dtype0
╓
!multi_head_attention/query/kernelVarHandleOp*
_output_shapes
: *2

debug_name$"multi_head_attention/query/kernel/*
dtype0*
shape:@@*2
shared_name#!multi_head_attention/query/kernel
Ы
5multi_head_attention/query/kernel/Read/ReadVariableOpReadVariableOp!multi_head_attention/query/kernel*"
_output_shapes
:@@*
dtype0
Ш
bve_layer2/biasVarHandleOp*
_output_shapes
: * 

debug_namebve_layer2/bias/*
dtype0*
shape:@* 
shared_namebve_layer2/bias
o
#bve_layer2/bias/Read/ReadVariableOpReadVariableOpbve_layer2/bias*
_output_shapes
:@*
dtype0
в
bve_layer2/kernelVarHandleOp*
_output_shapes
: *"

debug_namebve_layer2/kernel/*
dtype0*
shape
:@@*"
shared_namebve_layer2/kernel
w
%bve_layer2/kernel/Read/ReadVariableOpReadVariableOpbve_layer2/kernel*
_output_shapes

:@@*
dtype0
Ш
bve_layer1/biasVarHandleOp*
_output_shapes
: * 

debug_namebve_layer1/bias/*
dtype0*
shape:@* 
shared_namebve_layer1/bias
o
#bve_layer1/bias/Read/ReadVariableOpReadVariableOpbve_layer1/bias*
_output_shapes
:@*
dtype0
в
bve_layer1/kernelVarHandleOp*
_output_shapes
: *"

debug_namebve_layer1/kernel/*
dtype0*
shape
:@*"
shared_namebve_layer1/kernel
w
%bve_layer1/kernel/Read/ReadVariableOpReadVariableOpbve_layer1/kernel*
_output_shapes

:@*
dtype0
Ш
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
в
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
Ш
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
в
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
д
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
о
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
д
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
о
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
Т
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
Ь
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
П
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
Щ
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
П
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
Щ
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
П
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
Щ
dense_2/kernelVarHandleOp*
_output_shapes
: *

debug_namedense_2/kernel/*
dtype0*
shape
:@@*
shared_namedense_2/kernel
q
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel*
_output_shapes

:@@*
dtype0
П
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
Щ
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
Й

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
У
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
Ц
serving_default_Xbc_layerPlaceholder*4
_output_shapes"
 :                  *
dtype0*)
shape :                  
z
serving_default_t_inputPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
Ц
serving_default_tbc_layerPlaceholder*4
_output_shapes"
 :                  *
dtype0*)
shape :                  
Ц
serving_default_ubc_layerPlaceholder*4
_output_shapes"
 :                  *
dtype0*)
shape :                  
z
serving_default_x_inputPlaceholder*'
_output_shapes
:         *
dtype0*
shape:         
╪
StatefulPartitionedCallStatefulPartitionedCallserving_default_Xbc_layerserving_default_t_inputserving_default_tbc_layerserving_default_ubc_layerserving_default_x_inputbpe_layer1/kernelbpe_layer1/biasbpe_layer2/kernelbpe_layer2/biasbve_layer1/kernelbve_layer1/biasbve_layer2/kernelbve_layer2/biasspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/biasdense/kernel
dense/biasdense_1/kerneldense_1/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/bias#multi_head_attention_2/query/kernel!multi_head_attention_2/query/bias!multi_head_attention_2/key/kernelmulti_head_attention_2/key/bias#multi_head_attention_2/value/kernel!multi_head_attention_2/value/bias.multi_head_attention_2/attention_output/kernel,multi_head_attention_2/attention_output/biasdense_4/kerneldense_4/bias
ou1/kernelou1/bias
ou2/kernelou2/biasoutput_u/kerneloutput_u/bias*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8В *.
f)R'
%__inference_signature_wrapper_2999633

NoOpNoOp
Ъ└
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*╘┐
value╔┐B┼┐ B╜┐
Ю
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
layer-16
layer_with_weights-4
layer-17
layer_with_weights-5
layer-18
layer_with_weights-6
layer-19
layer-20
layer_with_weights-7
layer-21
layer_with_weights-8
layer-22
layer_with_weights-9
layer-23
layer-24
layer_with_weights-10
layer-25
layer-26
layer_with_weights-11
layer-27
layer_with_weights-12
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%
signatures*
* 
* 
О
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses* 
О
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses* 
* 
* 
О
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses* 
О
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses* 
О
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses* 
О
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses* 
О
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses* 
* 
°
Player_with_weights-0
Player-0
Qlayer-1
Rlayer_with_weights-1
Rlayer-2
Slayer-3
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses*
°
Zlayer_with_weights-0
Zlayer-0
[layer-1
\layer_with_weights-1
\layer-2
]layer-3
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses*
°
dlayer_with_weights-0
dlayer-0
elayer-1
flayer_with_weights-1
flayer-2
glayer-3
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses*
∙
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_query_dense
u
_key_dense
v_value_dense
w_softmax
x_dropout_layer
y_output_dense*
О
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses* 
о
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses
Жkernel
	Зbias*
о
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses
Оkernel
	Пbias*
Е
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses
Ц_query_dense
Ч
_key_dense
Ш_value_dense
Щ_softmax
Ъ_dropout_layer
Ы_output_dense*
Ф
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses* 
о
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses
иkernel
	йbias*
о
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
о__call__
+п&call_and_return_all_conditional_losses
░kernel
	▒bias*
Е
▓	variables
│trainable_variables
┤regularization_losses
╡	keras_api
╢__call__
+╖&call_and_return_all_conditional_losses
╕_query_dense
╣
_key_dense
║_value_dense
╗_softmax
╝_dropout_layer
╜_output_dense*
Ф
╛	variables
┐trainable_variables
└regularization_losses
┴	keras_api
┬__call__
+├&call_and_return_all_conditional_losses* 
о
─	variables
┼trainable_variables
╞regularization_losses
╟	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses
╩kernel
	╦bias*
Ф
╠	variables
═trainable_variables
╬regularization_losses
╧	keras_api
╨__call__
+╤&call_and_return_all_conditional_losses* 
Д
╥layer_with_weights-0
╥layer-0
╙layer-1
╘layer_with_weights-1
╘layer-2
╒layer-3
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses*
о
▄	variables
▌trainable_variables
▐regularization_losses
▀	keras_api
р__call__
+с&call_and_return_all_conditional_losses
тkernel
	уbias*
╬
ф0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
Ё12
ё13
Є14
є15
Ї16
ї17
Ў18
ў19
Ж20
З21
О22
П23
°24
∙25
·26
√27
№28
¤29
■30
 31
и32
й33
░34
▒35
А36
Б37
В38
Г39
Д40
Е41
Ж42
З43
╩44
╦45
И46
Й47
К48
Л49
т50
у51*
╬
ф0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
Ё12
ё13
Є14
є15
Ї16
ї17
Ў18
ў19
Ж20
З21
О22
П23
°24
∙25
·26
√27
№28
¤29
■30
 31
и32
й33
░34
▒35
А36
Б37
В38
Г39
Д40
Е41
Ж42
З43
╩44
╦45
И46
Й47
К48
Л49
т50
у51*
* 
╡
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses*

Сtrace_0
Тtrace_1* 

Уtrace_0
Фtrace_1* 
* 

Хserving_default* 
* 
* 
* 
Ц
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses* 

Ыtrace_0* 

Ьtrace_0* 
* 
* 
* 
Ц
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses* 

вtrace_0* 

гtrace_0* 
* 
* 
* 
Ц
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses* 

йtrace_0* 

кtrace_0* 
* 
* 
* 
Ц
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses* 

░trace_0* 

▒trace_0* 
* 
* 
* 
Ц
▓non_trainable_variables
│layers
┤metrics
 ╡layer_regularization_losses
╢layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses* 

╖trace_0* 

╕trace_0* 
* 
* 
* 
Ц
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses* 

╛trace_0* 

┐trace_0* 
* 
* 
* 
Ц
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses* 

┼trace_0* 

╞trace_0* 
о
╟	variables
╚trainable_variables
╔regularization_losses
╩	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses
фkernel
	хbias*
Ф
═	variables
╬trainable_variables
╧regularization_losses
╨	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses* 
о
╙	variables
╘trainable_variables
╒regularization_losses
╓	keras_api
╫__call__
+╪&call_and_return_all_conditional_losses
цkernel
	чbias*
Ф
┘	variables
┌trainable_variables
█regularization_losses
▄	keras_api
▌__call__
+▐&call_and_return_all_conditional_losses* 
$
ф0
х1
ц2
ч3*
$
ф0
х1
ц2
ч3*
* 
Ш
▀non_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses*

фtrace_0
хtrace_1* 

цtrace_0
чtrace_1* 
о
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses
шkernel
	щbias*
Ф
ю	variables
яtrainable_variables
Ёregularization_losses
ё	keras_api
Є__call__
+є&call_and_return_all_conditional_losses* 
о
Ї	variables
їtrainable_variables
Ўregularization_losses
ў	keras_api
°__call__
+∙&call_and_return_all_conditional_losses
ъkernel
	ыbias*
Ф
·	variables
√trainable_variables
№regularization_losses
¤	keras_api
■__call__
+ &call_and_return_all_conditional_losses* 
$
ш0
щ1
ъ2
ы3*
$
ш0
щ1
ъ2
ы3*
* 
Ш
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses*

Еtrace_0
Жtrace_1* 

Зtrace_0
Иtrace_1* 
о
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses
ьkernel
	эbias*
Ф
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses* 
о
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
юkernel
	яbias*
Ф
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses* 
$
ь0
э1
ю2
я3*
$
ь0
э1
ю2
я3*
* 
Ш
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses*

жtrace_0
зtrace_1* 

иtrace_0
йtrace_1* 
D
Ё0
ё1
Є2
є3
Ї4
ї5
Ў6
ў7*
D
Ё0
ё1
Є2
є3
Ї4
ї5
Ў6
ў7*
* 
Ш
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses*

пtrace_0
░trace_1* 

▒trace_0
▓trace_1* 
с
│	variables
┤trainable_variables
╡regularization_losses
╢	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses
╣partial_output_shape
║full_output_shape
Ёkernel
	ёbias*
с
╗	variables
╝trainable_variables
╜regularization_losses
╛	keras_api
┐__call__
+└&call_and_return_all_conditional_losses
┴partial_output_shape
┬full_output_shape
Єkernel
	єbias*
с
├	variables
─trainable_variables
┼regularization_losses
╞	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses
╔partial_output_shape
╩full_output_shape
Їkernel
	їbias*
Ф
╦	variables
╠trainable_variables
═regularization_losses
╬	keras_api
╧__call__
+╨&call_and_return_all_conditional_losses* 
м
╤	variables
╥trainable_variables
╙regularization_losses
╘	keras_api
╒__call__
+╓&call_and_return_all_conditional_losses
╫_random_generator* 
с
╪	variables
┘trainable_variables
┌regularization_losses
█	keras_api
▄__call__
+▌&call_and_return_all_conditional_losses
▐partial_output_shape
▀full_output_shape
Ўkernel
	ўbias*
* 
* 
* 
Ц
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses* 

хtrace_0* 

цtrace_0* 

Ж0
З1*

Ж0
З1*
* 
Ю
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses*

ьtrace_0* 

эtrace_0* 
\V
VARIABLE_VALUEdense/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE*
XR
VARIABLE_VALUE
dense/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE*

О0
П1*

О0
П1*
* 
Ю
юnon_trainable_variables
яlayers
Ёmetrics
 ёlayer_regularization_losses
Єlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses*

єtrace_0* 

Їtrace_0* 
^X
VARIABLE_VALUEdense_1/kernel6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_1/bias4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUE*
D
°0
∙1
·2
√3
№4
¤5
■6
 7*
D
°0
∙1
·2
√3
№4
¤5
■6
 7*
* 
Ю
їnon_trainable_variables
Ўlayers
ўmetrics
 °layer_regularization_losses
∙layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses*

·trace_0
√trace_1* 

№trace_0
¤trace_1* 
с
■	variables
 trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses
Дpartial_output_shape
Еfull_output_shape
°kernel
	∙bias*
с
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses
Мpartial_output_shape
Нfull_output_shape
·kernel
	√bias*
с
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses
Фpartial_output_shape
Хfull_output_shape
№kernel
	¤bias*
Ф
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses* 
м
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses
в_random_generator* 
с
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
з__call__
+и&call_and_return_all_conditional_losses
йpartial_output_shape
кfull_output_shape
■kernel
	 bias*
* 
* 
* 
Ь
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses* 

░trace_0* 

▒trace_0* 

и0
й1*

и0
й1*
* 
Ю
▓non_trainable_variables
│layers
┤metrics
 ╡layer_regularization_losses
╢layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses*

╖trace_0* 

╕trace_0* 
^X
VARIABLE_VALUEdense_2/kernel6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_2/bias4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUE*

░0
▒1*

░0
▒1*
* 
Ю
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
к	variables
лtrainable_variables
мregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses*

╛trace_0* 

┐trace_0* 
^X
VARIABLE_VALUEdense_3/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE*
ZT
VARIABLE_VALUEdense_3/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE*
D
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7*
D
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7*
* 
Ю
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
▓	variables
│trainable_variables
┤regularization_losses
╢__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses*

┼trace_0
╞trace_1* 

╟trace_0
╚trace_1* 
с
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
═__call__
+╬&call_and_return_all_conditional_losses
╧partial_output_shape
╨full_output_shape
Аkernel
	Бbias*
с
╤	variables
╥trainable_variables
╙regularization_losses
╘	keras_api
╒__call__
+╓&call_and_return_all_conditional_losses
╫partial_output_shape
╪full_output_shape
Вkernel
	Гbias*
с
┘	variables
┌trainable_variables
█regularization_losses
▄	keras_api
▌__call__
+▐&call_and_return_all_conditional_losses
▀partial_output_shape
рfull_output_shape
Дkernel
	Еbias*
Ф
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses* 
м
ч	variables
шtrainable_variables
щregularization_losses
ъ	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses
э_random_generator* 
с
ю	variables
яtrainable_variables
Ёregularization_losses
ё	keras_api
Є__call__
+є&call_and_return_all_conditional_losses
Їpartial_output_shape
їfull_output_shape
Жkernel
	Зbias*
* 
* 
* 
Ь
Ўnon_trainable_variables
ўlayers
°metrics
 ∙layer_regularization_losses
·layer_metrics
╛	variables
┐trainable_variables
└regularization_losses
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses* 

√trace_0* 

№trace_0* 

╩0
╦1*

╩0
╦1*
* 
Ю
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
─	variables
┼trainable_variables
╞regularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses*

Вtrace_0* 

Гtrace_0* 
_Y
VARIABLE_VALUEdense_4/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE*
[U
VARIABLE_VALUEdense_4/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE*
* 
* 
* 
Ь
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
╠	variables
═trainable_variables
╬regularization_losses
╨__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses* 

Йtrace_0* 

Кtrace_0* 
о
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses
Иkernel
	Йbias*
Ф
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses* 
о
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
Кkernel
	Лbias*
Ф
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses* 
$
И0
Й1
К2
Л3*
$
И0
Й1
К2
Л3*
* 
Ю
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses*

иtrace_0
йtrace_1* 

кtrace_0
лtrace_1* 

т0
у1*

т0
у1*
* 
Ю
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
▄	variables
▌trainable_variables
▐regularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses*

▒trace_0* 

▓trace_0* 
`Z
VARIABLE_VALUEoutput_u/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE*
\V
VARIABLE_VALUEoutput_u/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEspatial_layer1/kernel&variables/0/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEspatial_layer1/bias&variables/1/.ATTRIBUTES/VARIABLE_VALUE*
UO
VARIABLE_VALUEspatial_layer2/kernel&variables/2/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEspatial_layer2/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEbpe_layer1/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEbpe_layer1/bias&variables/5/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEbpe_layer2/kernel&variables/6/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEbpe_layer2/bias&variables/7/.ATTRIBUTES/VARIABLE_VALUE*
QK
VARIABLE_VALUEbve_layer1/kernel&variables/8/.ATTRIBUTES/VARIABLE_VALUE*
OI
VARIABLE_VALUEbve_layer1/bias&variables/9/.ATTRIBUTES/VARIABLE_VALUE*
RL
VARIABLE_VALUEbve_layer2/kernel'variables/10/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEbve_layer2/bias'variables/11/.ATTRIBUTES/VARIABLE_VALUE*
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
d^
VARIABLE_VALUE#multi_head_attention_1/query/kernel'variables/24/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/query/bias'variables/25/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/key/kernel'variables/26/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_1/key/bias'variables/27/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_1/value/kernel'variables/28/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_1/value/bias'variables/29/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_1/attention_output/kernel'variables/30/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_1/attention_output/bias'variables/31/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_2/query/kernel'variables/36/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/query/bias'variables/37/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/key/kernel'variables/38/.ATTRIBUTES/VARIABLE_VALUE*
`Z
VARIABLE_VALUEmulti_head_attention_2/key/bias'variables/39/.ATTRIBUTES/VARIABLE_VALUE*
d^
VARIABLE_VALUE#multi_head_attention_2/value/kernel'variables/40/.ATTRIBUTES/VARIABLE_VALUE*
b\
VARIABLE_VALUE!multi_head_attention_2/value/bias'variables/41/.ATTRIBUTES/VARIABLE_VALUE*
oi
VARIABLE_VALUE.multi_head_attention_2/attention_output/kernel'variables/42/.ATTRIBUTES/VARIABLE_VALUE*
mg
VARIABLE_VALUE,multi_head_attention_2/attention_output/bias'variables/43/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
ou1/kernel'variables/46/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEou1/bias'variables/47/.ATTRIBUTES/VARIABLE_VALUE*
KE
VARIABLE_VALUE
ou2/kernel'variables/48/.ATTRIBUTES/VARIABLE_VALUE*
IC
VARIABLE_VALUEou2/bias'variables/49/.ATTRIBUTES/VARIABLE_VALUE*
* 
т
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
28*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
ф0
х1*

ф0
х1*
* 
Ю
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
╟	variables
╚trainable_variables
╔regularization_losses
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses*

╕trace_0* 

╣trace_0* 
* 
* 
* 
Ь
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
═	variables
╬trainable_variables
╧regularization_losses
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses* 

┐trace_0* 

└trace_0* 

ц0
ч1*

ц0
ч1*
* 
Ю
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
╙	variables
╘trainable_variables
╒regularization_losses
╫__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses*

╞trace_0* 

╟trace_0* 
* 
* 
* 
Ь
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
┘	variables
┌trainable_variables
█regularization_losses
▌__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses* 

═trace_0* 

╬trace_0* 
* 
 
P0
Q1
R2
S3*
* 
* 
* 
* 
* 
* 
* 

ш0
щ1*

ш0
щ1*
* 
Ю
╧non_trainable_variables
╨layers
╤metrics
 ╥layer_regularization_losses
╙layer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses*

╘trace_0* 

╒trace_0* 
* 
* 
* 
Ь
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
ю	variables
яtrainable_variables
Ёregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses* 

█trace_0* 

▄trace_0* 

ъ0
ы1*

ъ0
ы1*
* 
Ю
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
Ї	variables
їtrainable_variables
Ўregularization_losses
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses*

тtrace_0* 

уtrace_0* 
* 
* 
* 
Ь
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
·	variables
√trainable_variables
№regularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses* 

щtrace_0* 

ъtrace_0* 
* 
 
Z0
[1
\2
]3*
* 
* 
* 
* 
* 
* 
* 

ь0
э1*

ь0
э1*
* 
Ю
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses*

Ёtrace_0* 

ёtrace_0* 
* 
* 
* 
Ь
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses* 

ўtrace_0* 

°trace_0* 

ю0
я1*

ю0
я1*
* 
Ю
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses*

■trace_0* 

 trace_0* 
* 
* 
* 
Ь
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses* 

Еtrace_0* 

Жtrace_0* 
* 
 
d0
e1
f2
g3*
* 
* 
* 
* 
* 
* 
* 
* 
.
t0
u1
v2
w3
x4
y5*
* 
* 
* 
* 
* 
* 
* 

Ё0
ё1*

Ё0
ё1*
* 
Ю
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
│	variables
┤trainable_variables
╡regularization_losses
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses*
* 
* 
* 
* 

Є0
є1*

Є0
є1*
* 
Ю
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
╗	variables
╝trainable_variables
╜regularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses*
* 
* 
* 
* 

Ї0
ї1*

Ї0
ї1*
* 
Ю
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
├	variables
─trainable_variables
┼regularization_losses
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
Ь
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
╦	variables
╠trainable_variables
═regularization_losses
╧__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
╤	variables
╥trainable_variables
╙regularization_losses
╒__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses* 
* 
* 
* 

Ў0
ў1*

Ў0
ў1*
* 
Ю
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
╪	variables
┘trainable_variables
┌regularization_losses
▄__call__
+▌&call_and_return_all_conditional_losses
'▌"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
Ц0
Ч1
Ш2
Щ3
Ъ4
Ы5*
* 
* 
* 
* 
* 
* 
* 

°0
∙1*

°0
∙1*
* 
Ю
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
■	variables
 trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses*
* 
* 
* 
* 

·0
√1*

·0
√1*
* 
Ю
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses*
* 
* 
* 
* 

№0
¤1*

№0
¤1*
* 
Ю
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
Ь
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses* 
* 
* 
* 

■0
 1*

■0
 1*
* 
Ю
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
г	variables
дtrainable_variables
еregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
4
╕0
╣1
║2
╗3
╝4
╜5*
* 
* 
* 
* 
* 
* 
* 

А0
Б1*

А0
Б1*
* 
Ю
├non_trainable_variables
─layers
┼metrics
 ╞layer_regularization_losses
╟layer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses*
* 
* 
* 
* 

В0
Г1*

В0
Г1*
* 
Ю
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
╤	variables
╥trainable_variables
╙regularization_losses
╒__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses*
* 
* 
* 
* 

Д0
Е1*

Д0
Е1*
* 
Ю
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
┘	variables
┌trainable_variables
█regularization_losses
▌__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
Ь
╥non_trainable_variables
╙layers
╘metrics
 ╒layer_regularization_losses
╓layer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses* 
* 
* 
* 
* 
* 
Ь
╫non_trainable_variables
╪layers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
ч	variables
шtrainable_variables
щregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses* 
* 
* 
* 

Ж0
З1*

Ж0
З1*
* 
Ю
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
ю	variables
яtrainable_variables
Ёregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
И0
Й1*

И0
Й1*
* 
Ю
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses*

цtrace_0* 

чtrace_0* 
* 
* 
* 
Ь
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses* 

эtrace_0* 

юtrace_0* 

К0
Л1*

К0
Л1*
* 
Ю
яnon_trainable_variables
Ёlayers
ёmetrics
 Єlayer_regularization_losses
єlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses*

Їtrace_0* 

їtrace_0* 
* 
* 
* 
Ь
Ўnon_trainable_variables
ўlayers
°metrics
 ∙layer_regularization_losses
·layer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses* 

√trace_0* 

№trace_0* 
* 
$
╥0
╙1
╘2
╒3*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
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
Я
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasoutput_u/kerneloutput_u/biasspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/biasbpe_layer1/kernelbpe_layer1/biasbpe_layer2/kernelbpe_layer2/biasbve_layer1/kernelbve_layer1/biasbve_layer2/kernelbve_layer2/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/bias#multi_head_attention_2/query/kernel!multi_head_attention_2/query/bias!multi_head_attention_2/key/kernelmulti_head_attention_2/key/bias#multi_head_attention_2/value/kernel!multi_head_attention_2/value/bias.multi_head_attention_2/attention_output/kernel,multi_head_attention_2/attention_output/bias
ou1/kernelou1/bias
ou2/kernelou2/biasConst*A
Tin:
826*
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
GPU2*0J 8В *)
f$R"
 __inference__traced_save_3001039
Ъ
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasoutput_u/kerneloutput_u/biasspatial_layer1/kernelspatial_layer1/biasspatial_layer2/kernelspatial_layer2/biasbpe_layer1/kernelbpe_layer1/biasbpe_layer2/kernelbpe_layer2/biasbve_layer1/kernelbve_layer1/biasbve_layer2/kernelbve_layer2/bias!multi_head_attention/query/kernelmulti_head_attention/query/biasmulti_head_attention/key/kernelmulti_head_attention/key/bias!multi_head_attention/value/kernelmulti_head_attention/value/bias,multi_head_attention/attention_output/kernel*multi_head_attention/attention_output/bias#multi_head_attention_1/query/kernel!multi_head_attention_1/query/bias!multi_head_attention_1/key/kernelmulti_head_attention_1/key/bias#multi_head_attention_1/value/kernel!multi_head_attention_1/value/bias.multi_head_attention_1/attention_output/kernel,multi_head_attention_1/attention_output/bias#multi_head_attention_2/query/kernel!multi_head_attention_2/query/bias!multi_head_attention_2/key/kernelmulti_head_attention_2/key/bias#multi_head_attention_2/value/kernel!multi_head_attention_2/value/bias.multi_head_attention_2/attention_output/kernel,multi_head_attention_2/attention_output/bias
ou1/kernelou1/bias
ou2/kernelou2/bias*@
Tin9
725*
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
GPU2*0J 8В *,
f'R%
#__inference__traced_restore_3001204╘б
╜.
Г
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2998440	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
С
Э
0__inference_spatial_layer1_layer_call_fn_3000358

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_2997753s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000354:'#
!
_user_specified_name	3000352:S O
+
_output_shapes
:         
 
_user_specified_nameinputs
╫
√
D__inference_dense_4_layer_call_and_return_conditional_losses_3000319

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
├
e
I__inference_activation_7_layer_call_and_return_conditional_losses_2998232

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:         @P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
Т
v
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2999729
inputs_0
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :Д
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  d
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:                  :                  :^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
у	
╥
%__inference_BVE_layer_call_fn_2998148
bve_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallbve_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_BVE_layer_call_and_return_conditional_losses_2998119|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2998144:'#
!
_user_specified_name	2998142:'#
!
_user_specified_name	2998140:'#
!
_user_specified_name	2998138:f b
4
_output_shapes"
 :                  
*
_user_specified_namebve_layer1_input
╫
√
D__inference_dense_1_layer_call_and_return_conditional_losses_2999939

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
э

`
D__inference_reshape_layer_call_and_return_conditional_losses_2999716

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
         П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         \
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╚
d
H__inference_rescaling_1_layer_call_and_return_conditional_losses_2998328

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
 *  А┐U
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:         Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╡
Q
%__inference_add_layer_call_fn_2999853
inputs_0
inputs_1
identity┐
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_2998463d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:UQ
+
_output_shapes
:         @
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:         @
"
_user_specified_name
inputs_0
▐
J
.__inference_activation_4_layer_call_fn_3000589

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_2998075m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╒
∙
B__inference_dense_layer_call_and_return_conditional_losses_2999899

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
Й
d
H__inference_rescaling_3_layer_call_and_return_conditional_losses_2999698

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
 *  А┐b
mulMulinputsCast/x:output:0*
T0*4
_output_shapes"
 :                  g
addAddV2mul:z:0Cast_1/x:output:0*
T0*4
_output_shapes"
 :                  \
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
°	
Ў
E__inference_output_u_layer_call_and_return_conditional_losses_3000349

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
д
G
+__inference_rescaling_layer_call_fn_2999638

inputs
identity┤
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_2998319`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
┐.
Е
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_2998937	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
╞
Є
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997823
spatial_layer1_input(
spatial_layer1_2997810:@$
spatial_layer1_2997812:@(
spatial_layer2_2997816:@@$
spatial_layer2_2997818:@
identityИв&spatial_layer1/StatefulPartitionedCallв&spatial_layer2/StatefulPartitionedCallа
&spatial_layer1/StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputspatial_layer1_2997810spatial_layer1_2997812*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_2997753э
activation/PartitionedCallPartitionedCall/spatial_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_2997763п
&spatial_layer2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0spatial_layer2_2997816spatial_layer2_2997818*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_2997794ё
activation_1/PartitionedCallPartitionedCall/spatial_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_2997804x
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @t
NoOpNoOp'^spatial_layer1/StatefulPartitionedCall'^spatial_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : 2P
&spatial_layer1/StatefulPartitionedCall&spatial_layer1/StatefulPartitionedCall2P
&spatial_layer2/StatefulPartitionedCall&spatial_layer2/StatefulPartitionedCall:'#
!
_user_specified_name	2997818:'#
!
_user_specified_name	2997816:'#
!
_user_specified_name	2997812:'#
!
_user_specified_name	2997810:a ]
+
_output_shapes
:         
.
_user_specified_namespatial_layer1_input
и
E
)__inference_reshape_layer_call_fn_2999703

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2998376d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
и
I
-__inference_rescaling_1_layer_call_fn_2999651

inputs
identity╢
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_2998328`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
н
Щ
,__inference_bpe_layer1_layer_call_fn_3000456

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bpe_layer1_layer_call_and_return_conditional_losses_2997909|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000452:'#
!
_user_specified_name	3000450:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
у	
╥
%__inference_BVE_layer_call_fn_2998161
bve_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallbve_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_BVE_layer_call_and_return_conditional_losses_2998135|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2998157:'#
!
_user_specified_name	2998155:'#
!
_user_specified_name	2998153:'#
!
_user_specified_name	2998151:f b
4
_output_shapes"
 :                  
*
_user_specified_namebve_layer1_input
Р
В
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_2997794

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
ў
e
I__inference_activation_2_layer_call_and_return_conditional_losses_2997919

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :                  @]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
┐.
Е
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_3000231	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
┬
Ї
6__inference_multi_head_attention_layer_call_fn_2999775	
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
identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2998873s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	2999771:'	#
!
_user_specified_name	2999769:'#
!
_user_specified_name	2999767:'#
!
_user_specified_name	2999765:'#
!
_user_specified_name	2999763:'#
!
_user_specified_name	2999761:'#
!
_user_specified_name	2999759:'#
!
_user_specified_name	2999757:YU
4
_output_shapes"
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
┐.
Е
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_2998704	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
э

`
D__inference_reshape_layer_call_and_return_conditional_losses_2998376

inputs
identityI
ShapeShapeinputs*
T0*
_output_shapes
::э╧]
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
valueB:╤
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
         П
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:h
ReshapeReshapeinputsReshape/shape:output:0*
T0*+
_output_shapes
:         \
IdentityIdentityReshape:output:0*
T0*+
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
╧
l
B__inference_add_2_layer_call_and_return_conditional_losses_2998727

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:         @S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:SO
+
_output_shapes
:         @
 
_user_specified_nameinputs:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╫
n
B__inference_add_1_layer_call_and_return_conditional_losses_3000069
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:         @S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:UQ
+
_output_shapes
:         @
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:         @
"
_user_specified_name
inputs_0
ба
ц6
"__inference__wrapped_model_2997721
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layerH
6model_bpe_bpe_layer1_tensordot_readvariableop_resource:@B
4model_bpe_bpe_layer1_biasadd_readvariableop_resource:@H
6model_bpe_bpe_layer2_tensordot_readvariableop_resource:@@B
4model_bpe_bpe_layer2_biasadd_readvariableop_resource:@H
6model_bve_bve_layer1_tensordot_readvariableop_resource:@B
4model_bve_bve_layer1_biasadd_readvariableop_resource:@H
6model_bve_bve_layer2_tensordot_readvariableop_resource:@@B
4model_bve_bve_layer2_biasadd_readvariableop_resource:@_
Mmodel_spatial_transformation_spatial_layer1_tensordot_readvariableop_resource:@Y
Kmodel_spatial_transformation_spatial_layer1_biasadd_readvariableop_resource:@_
Mmodel_spatial_transformation_spatial_layer2_tensordot_readvariableop_resource:@@Y
Kmodel_spatial_transformation_spatial_layer2_biasadd_readvariableop_resource:@\
Fmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource:@@N
<model_multi_head_attention_query_add_readvariableop_resource:@Z
Dmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource:@@L
:model_multi_head_attention_key_add_readvariableop_resource:@\
Fmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource:@@N
<model_multi_head_attention_value_add_readvariableop_resource:@g
Qmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource:@@U
Gmodel_multi_head_attention_attention_output_add_readvariableop_resource:@?
-model_dense_tensordot_readvariableop_resource:@@9
+model_dense_biasadd_readvariableop_resource:@A
/model_dense_1_tensordot_readvariableop_resource:@@;
-model_dense_1_biasadd_readvariableop_resource:@^
Hmodel_multi_head_attention_1_query_einsum_einsum_readvariableop_resource:@@P
>model_multi_head_attention_1_query_add_readvariableop_resource:@\
Fmodel_multi_head_attention_1_key_einsum_einsum_readvariableop_resource:@@N
<model_multi_head_attention_1_key_add_readvariableop_resource:@^
Hmodel_multi_head_attention_1_value_einsum_einsum_readvariableop_resource:@@P
>model_multi_head_attention_1_value_add_readvariableop_resource:@i
Smodel_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource:@@W
Imodel_multi_head_attention_1_attention_output_add_readvariableop_resource:@A
/model_dense_2_tensordot_readvariableop_resource:@@;
-model_dense_2_biasadd_readvariableop_resource:@A
/model_dense_3_tensordot_readvariableop_resource:@@;
-model_dense_3_biasadd_readvariableop_resource:@^
Hmodel_multi_head_attention_2_query_einsum_einsum_readvariableop_resource:@@P
>model_multi_head_attention_2_query_add_readvariableop_resource:@\
Fmodel_multi_head_attention_2_key_einsum_einsum_readvariableop_resource:@@N
<model_multi_head_attention_2_key_add_readvariableop_resource:@^
Hmodel_multi_head_attention_2_value_einsum_einsum_readvariableop_resource:@@P
>model_multi_head_attention_2_value_add_readvariableop_resource:@i
Smodel_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource:@@W
Imodel_multi_head_attention_2_attention_output_add_readvariableop_resource:@A
/model_dense_4_tensordot_readvariableop_resource:@@;
-model_dense_4_biasadd_readvariableop_resource:@<
*model_u_ou1_matmul_readvariableop_resource:@@9
+model_u_ou1_biasadd_readvariableop_resource:@<
*model_u_ou2_matmul_readvariableop_resource:@@9
+model_u_ou2_biasadd_readvariableop_resource:@?
-model_output_u_matmul_readvariableop_resource:@<
.model_output_u_biasadd_readvariableop_resource:
identityИв+model/BPE/bpe_layer1/BiasAdd/ReadVariableOpв-model/BPE/bpe_layer1/Tensordot/ReadVariableOpв+model/BPE/bpe_layer2/BiasAdd/ReadVariableOpв-model/BPE/bpe_layer2/Tensordot/ReadVariableOpв+model/BVE/bve_layer1/BiasAdd/ReadVariableOpв-model/BVE/bve_layer1/Tensordot/ReadVariableOpв+model/BVE/bve_layer2/BiasAdd/ReadVariableOpв-model/BVE/bve_layer2/Tensordot/ReadVariableOpв"model/U/ou1/BiasAdd/ReadVariableOpв!model/U/ou1/MatMul/ReadVariableOpв"model/U/ou2/BiasAdd/ReadVariableOpв!model/U/ou2/MatMul/ReadVariableOpв"model/dense/BiasAdd/ReadVariableOpв$model/dense/Tensordot/ReadVariableOpв$model/dense_1/BiasAdd/ReadVariableOpв&model/dense_1/Tensordot/ReadVariableOpв$model/dense_2/BiasAdd/ReadVariableOpв&model/dense_2/Tensordot/ReadVariableOpв$model/dense_3/BiasAdd/ReadVariableOpв&model/dense_3/Tensordot/ReadVariableOpв$model/dense_4/BiasAdd/ReadVariableOpв&model/dense_4/Tensordot/ReadVariableOpв>model/multi_head_attention/attention_output/add/ReadVariableOpвHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpв1model/multi_head_attention/key/add/ReadVariableOpв;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpв3model/multi_head_attention/query/add/ReadVariableOpв=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpв3model/multi_head_attention/value/add/ReadVariableOpв=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpв@model/multi_head_attention_1/attention_output/add/ReadVariableOpвJmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpв3model/multi_head_attention_1/key/add/ReadVariableOpв=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpв5model/multi_head_attention_1/query/add/ReadVariableOpв?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpв5model/multi_head_attention_1/value/add/ReadVariableOpв?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpв@model/multi_head_attention_2/attention_output/add/ReadVariableOpвJmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpв3model/multi_head_attention_2/key/add/ReadVariableOpв=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpв5model/multi_head_attention_2/query/add/ReadVariableOpв?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpв5model/multi_head_attention_2/value/add/ReadVariableOpв?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpв%model/output_u/BiasAdd/ReadVariableOpв$model/output_u/MatMul/ReadVariableOpвBmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpвDmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpвBmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpвDmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp[
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
 *  А┐v
model/rescaling/mulMulx_inputmodel/rescaling/Cast/x:output:0*
T0*'
_output_shapes
:         К
model/rescaling/addAddV2model/rescaling/mul:z:0!model/rescaling/Cast_1/x:output:0*
T0*'
_output_shapes
:         ]
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
 *  А┐z
model/rescaling_1/mulMult_input!model/rescaling_1/Cast/x:output:0*
T0*'
_output_shapes
:         Р
model/rescaling_1/addAddV2model/rescaling_1/mul:z:0#model/rescaling_1/Cast_1/x:output:0*
T0*'
_output_shapes
:         ]
model/rescaling_2/Cast/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @_
model/rescaling_2/Cast_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *  А┐Й
model/rescaling_2/mulMul	xbc_layer!model/rescaling_2/Cast/x:output:0*
T0*4
_output_shapes"
 :                  Э
model/rescaling_2/addAddV2model/rescaling_2/mul:z:0#model/rescaling_2/Cast_1/x:output:0*
T0*4
_output_shapes"
 :                  ]
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
 *  А┐Й
model/rescaling_3/mulMul	tbc_layer!model/rescaling_3/Cast/x:output:0*
T0*4
_output_shapes"
 :                  Э
model/rescaling_3/addAddV2model/rescaling_3/mul:z:0#model/rescaling_3/Cast_1/x:output:0*
T0*4
_output_shapes"
 :                  _
model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╗
model/concatenate/concatConcatV2model/rescaling/add:z:0model/rescaling_1/add:z:0&model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:         a
model/concatenate_1/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :╬
model/concatenate_1/concatConcatV2model/rescaling_2/add:z:0model/rescaling_3/add:z:0(model/concatenate_1/concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  r
model/reshape/ShapeShape!model/concatenate/concat:output:0*
T0*
_output_shapes
::э╧k
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
valueB:Ч
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
         ╟
model/reshape/Reshape/shapePack$model/reshape/strided_slice:output:0&model/reshape/Reshape/shape/1:output:0&model/reshape/Reshape/shape/2:output:0*
N*
T0*
_output_shapes
:Я
model/reshape/ReshapeReshape!model/concatenate/concat:output:0$model/reshape/Reshape/shape:output:0*
T0*+
_output_shapes
:         д
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
valueB"       Е
$model/BPE/bpe_layer1/Tensordot/ShapeShape#model/concatenate_1/concat:output:0*
T0*
_output_shapes
::э╧n
,model/BPE/bpe_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
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
value	B : У
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
valueB: н
#model/BPE/bpe_layer1/Tensordot/ProdProd0model/BPE/bpe_layer1/Tensordot/GatherV2:output:0-model/BPE/bpe_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&model/BPE/bpe_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%model/BPE/bpe_layer1/Tensordot/Prod_1Prod2model/BPE/bpe_layer1/Tensordot/GatherV2_1:output:0/model/BPE/bpe_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*model/BPE/bpe_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%model/BPE/bpe_layer1/Tensordot/concatConcatV2,model/BPE/bpe_layer1/Tensordot/free:output:0,model/BPE/bpe_layer1/Tensordot/axes:output:03model/BPE/bpe_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$model/BPE/bpe_layer1/Tensordot/stackPack,model/BPE/bpe_layer1/Tensordot/Prod:output:0.model/BPE/bpe_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:╔
(model/BPE/bpe_layer1/Tensordot/transpose	Transpose#model/concatenate_1/concat:output:0.model/BPE/bpe_layer1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  ╔
&model/BPE/bpe_layer1/Tensordot/ReshapeReshape,model/BPE/bpe_layer1/Tensordot/transpose:y:0-model/BPE/bpe_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%model/BPE/bpe_layer1/Tensordot/MatMulMatMul/model/BPE/bpe_layer1/Tensordot/Reshape:output:05model/BPE/bpe_layer1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
&model/BPE/bpe_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,model/BPE/bpe_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'model/BPE/bpe_layer1/Tensordot/concat_1ConcatV20model/BPE/bpe_layer1/Tensordot/GatherV2:output:0/model/BPE/bpe_layer1/Tensordot/Const_2:output:05model/BPE/bpe_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╦
model/BPE/bpe_layer1/TensordotReshape/model/BPE/bpe_layer1/Tensordot/MatMul:product:00model/BPE/bpe_layer1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Ь
+model/BPE/bpe_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_bpe_bpe_layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0─
model/BPE/bpe_layer1/BiasAddBiasAdd'model/BPE/bpe_layer1/Tensordot:output:03model/BPE/bpe_layer1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Й
model/BPE/activation_2/TanhTanh%model/BPE/bpe_layer1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @д
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
valueB"       Б
$model/BPE/bpe_layer2/Tensordot/ShapeShapemodel/BPE/activation_2/Tanh:y:0*
T0*
_output_shapes
::э╧n
,model/BPE/bpe_layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
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
value	B : У
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
valueB: н
#model/BPE/bpe_layer2/Tensordot/ProdProd0model/BPE/bpe_layer2/Tensordot/GatherV2:output:0-model/BPE/bpe_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&model/BPE/bpe_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%model/BPE/bpe_layer2/Tensordot/Prod_1Prod2model/BPE/bpe_layer2/Tensordot/GatherV2_1:output:0/model/BPE/bpe_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*model/BPE/bpe_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%model/BPE/bpe_layer2/Tensordot/concatConcatV2,model/BPE/bpe_layer2/Tensordot/free:output:0,model/BPE/bpe_layer2/Tensordot/axes:output:03model/BPE/bpe_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$model/BPE/bpe_layer2/Tensordot/stackPack,model/BPE/bpe_layer2/Tensordot/Prod:output:0.model/BPE/bpe_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┼
(model/BPE/bpe_layer2/Tensordot/transpose	Transposemodel/BPE/activation_2/Tanh:y:0.model/BPE/bpe_layer2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @╔
&model/BPE/bpe_layer2/Tensordot/ReshapeReshape,model/BPE/bpe_layer2/Tensordot/transpose:y:0-model/BPE/bpe_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%model/BPE/bpe_layer2/Tensordot/MatMulMatMul/model/BPE/bpe_layer2/Tensordot/Reshape:output:05model/BPE/bpe_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
&model/BPE/bpe_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,model/BPE/bpe_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'model/BPE/bpe_layer2/Tensordot/concat_1ConcatV20model/BPE/bpe_layer2/Tensordot/GatherV2:output:0/model/BPE/bpe_layer2/Tensordot/Const_2:output:05model/BPE/bpe_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╦
model/BPE/bpe_layer2/TensordotReshape/model/BPE/bpe_layer2/Tensordot/MatMul:product:00model/BPE/bpe_layer2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Ь
+model/BPE/bpe_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_bpe_bpe_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0─
model/BPE/bpe_layer2/BiasAddBiasAdd'model/BPE/bpe_layer2/Tensordot:output:03model/BPE/bpe_layer2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Й
model/BPE/activation_3/TanhTanh%model/BPE/bpe_layer2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @д
-model/BVE/bve_layer1/Tensordot/ReadVariableOpReadVariableOp6model_bve_bve_layer1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0m
#model/BVE/bve_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#model/BVE/bve_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       k
$model/BVE/bve_layer1/Tensordot/ShapeShape	ubc_layer*
T0*
_output_shapes
::э╧n
,model/BVE/bve_layer1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'model/BVE/bve_layer1/Tensordot/GatherV2GatherV2-model/BVE/bve_layer1/Tensordot/Shape:output:0,model/BVE/bve_layer1/Tensordot/free:output:05model/BVE/bve_layer1/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.model/BVE/bve_layer1/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)model/BVE/bve_layer1/Tensordot/GatherV2_1GatherV2-model/BVE/bve_layer1/Tensordot/Shape:output:0,model/BVE/bve_layer1/Tensordot/axes:output:07model/BVE/bve_layer1/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$model/BVE/bve_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
#model/BVE/bve_layer1/Tensordot/ProdProd0model/BVE/bve_layer1/Tensordot/GatherV2:output:0-model/BVE/bve_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&model/BVE/bve_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%model/BVE/bve_layer1/Tensordot/Prod_1Prod2model/BVE/bve_layer1/Tensordot/GatherV2_1:output:0/model/BVE/bve_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*model/BVE/bve_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%model/BVE/bve_layer1/Tensordot/concatConcatV2,model/BVE/bve_layer1/Tensordot/free:output:0,model/BVE/bve_layer1/Tensordot/axes:output:03model/BVE/bve_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$model/BVE/bve_layer1/Tensordot/stackPack,model/BVE/bve_layer1/Tensordot/Prod:output:0.model/BVE/bve_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:п
(model/BVE/bve_layer1/Tensordot/transpose	Transpose	ubc_layer.model/BVE/bve_layer1/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  ╔
&model/BVE/bve_layer1/Tensordot/ReshapeReshape,model/BVE/bve_layer1/Tensordot/transpose:y:0-model/BVE/bve_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%model/BVE/bve_layer1/Tensordot/MatMulMatMul/model/BVE/bve_layer1/Tensordot/Reshape:output:05model/BVE/bve_layer1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
&model/BVE/bve_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,model/BVE/bve_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'model/BVE/bve_layer1/Tensordot/concat_1ConcatV20model/BVE/bve_layer1/Tensordot/GatherV2:output:0/model/BVE/bve_layer1/Tensordot/Const_2:output:05model/BVE/bve_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╦
model/BVE/bve_layer1/TensordotReshape/model/BVE/bve_layer1/Tensordot/MatMul:product:00model/BVE/bve_layer1/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Ь
+model/BVE/bve_layer1/BiasAdd/ReadVariableOpReadVariableOp4model_bve_bve_layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0─
model/BVE/bve_layer1/BiasAddBiasAdd'model/BVE/bve_layer1/Tensordot:output:03model/BVE/bve_layer1/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Й
model/BVE/activation_4/TanhTanh%model/BVE/bve_layer1/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @д
-model/BVE/bve_layer2/Tensordot/ReadVariableOpReadVariableOp6model_bve_bve_layer2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0m
#model/BVE/bve_layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:t
#model/BVE/bve_layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Б
$model/BVE/bve_layer2/Tensordot/ShapeShapemodel/BVE/activation_4/Tanh:y:0*
T0*
_output_shapes
::э╧n
,model/BVE/bve_layer2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : П
'model/BVE/bve_layer2/Tensordot/GatherV2GatherV2-model/BVE/bve_layer2/Tensordot/Shape:output:0,model/BVE/bve_layer2/Tensordot/free:output:05model/BVE/bve_layer2/Tensordot/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:p
.model/BVE/bve_layer2/Tensordot/GatherV2_1/axisConst*
_output_shapes
: *
dtype0*
value	B : У
)model/BVE/bve_layer2/Tensordot/GatherV2_1GatherV2-model/BVE/bve_layer2/Tensordot/Shape:output:0,model/BVE/bve_layer2/Tensordot/axes:output:07model/BVE/bve_layer2/Tensordot/GatherV2_1/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
:n
$model/BVE/bve_layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: н
#model/BVE/bve_layer2/Tensordot/ProdProd0model/BVE/bve_layer2/Tensordot/GatherV2:output:0-model/BVE/bve_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: p
&model/BVE/bve_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: │
%model/BVE/bve_layer2/Tensordot/Prod_1Prod2model/BVE/bve_layer2/Tensordot/GatherV2_1:output:0/model/BVE/bve_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: l
*model/BVE/bve_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : Ё
%model/BVE/bve_layer2/Tensordot/concatConcatV2,model/BVE/bve_layer2/Tensordot/free:output:0,model/BVE/bve_layer2/Tensordot/axes:output:03model/BVE/bve_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:╕
$model/BVE/bve_layer2/Tensordot/stackPack,model/BVE/bve_layer2/Tensordot/Prod:output:0.model/BVE/bve_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:┼
(model/BVE/bve_layer2/Tensordot/transpose	Transposemodel/BVE/activation_4/Tanh:y:0.model/BVE/bve_layer2/Tensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @╔
&model/BVE/bve_layer2/Tensordot/ReshapeReshape,model/BVE/bve_layer2/Tensordot/transpose:y:0-model/BVE/bve_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ╔
%model/BVE/bve_layer2/Tensordot/MatMulMatMul/model/BVE/bve_layer2/Tensordot/Reshape:output:05model/BVE/bve_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @p
&model/BVE/bve_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@n
,model/BVE/bve_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : √
'model/BVE/bve_layer2/Tensordot/concat_1ConcatV20model/BVE/bve_layer2/Tensordot/GatherV2:output:0/model/BVE/bve_layer2/Tensordot/Const_2:output:05model/BVE/bve_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:╦
model/BVE/bve_layer2/TensordotReshape/model/BVE/bve_layer2/Tensordot/MatMul:product:00model/BVE/bve_layer2/Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @Ь
+model/BVE/bve_layer2/BiasAdd/ReadVariableOpReadVariableOp4model_bve_bve_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0─
model/BVE/bve_layer2/BiasAddBiasAdd'model/BVE/bve_layer2/Tensordot:output:03model/BVE/bve_layer2/BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @Й
model/BVE/activation_5/TanhTanh%model/BVE/bve_layer2/BiasAdd:output:0*
T0*4
_output_shapes"
 :                  @╥
Dmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpReadVariableOpMmodel_spatial_transformation_spatial_layer1_tensordot_readvariableop_resource*
_output_shapes

:@*
dtype0Д
:model/spatial_transformation/spatial_layer1/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Л
:model/spatial_transformation/spatial_layer1/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       Ч
;model/spatial_transformation/spatial_layer1/Tensordot/ShapeShapemodel/reshape/Reshape:output:0*
T0*
_output_shapes
::э╧Е
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
:З
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
:Е
;model/spatial_transformation/spatial_layer1/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Є
:model/spatial_transformation/spatial_layer1/Tensordot/ProdProdGmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2:output:0Dmodel/spatial_transformation/spatial_layer1/Tensordot/Const:output:0*
T0*
_output_shapes
: З
=model/spatial_transformation/spatial_layer1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
<model/spatial_transformation/spatial_layer1/Tensordot/Prod_1ProdImodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2_1:output:0Fmodel/spatial_transformation/spatial_layer1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Г
Amodel/spatial_transformation/spatial_layer1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╠
<model/spatial_transformation/spatial_layer1/Tensordot/concatConcatV2Cmodel/spatial_transformation/spatial_layer1/Tensordot/free:output:0Cmodel/spatial_transformation/spatial_layer1/Tensordot/axes:output:0Jmodel/spatial_transformation/spatial_layer1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
;model/spatial_transformation/spatial_layer1/Tensordot/stackPackCmodel/spatial_transformation/spatial_layer1/Tensordot/Prod:output:0Emodel/spatial_transformation/spatial_layer1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:щ
?model/spatial_transformation/spatial_layer1/Tensordot/transpose	Transposemodel/reshape/Reshape:output:0Emodel/spatial_transformation/spatial_layer1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         О
=model/spatial_transformation/spatial_layer1/Tensordot/ReshapeReshapeCmodel/spatial_transformation/spatial_layer1/Tensordot/transpose:y:0Dmodel/spatial_transformation/spatial_layer1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  О
<model/spatial_transformation/spatial_layer1/Tensordot/MatMulMatMulFmodel/spatial_transformation/spatial_layer1/Tensordot/Reshape:output:0Lmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @З
=model/spatial_transformation/spatial_layer1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Е
Cmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
>model/spatial_transformation/spatial_layer1/Tensordot/concat_1ConcatV2Gmodel/spatial_transformation/spatial_layer1/Tensordot/GatherV2:output:0Fmodel/spatial_transformation/spatial_layer1/Tensordot/Const_2:output:0Lmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:З
5model/spatial_transformation/spatial_layer1/TensordotReshapeFmodel/spatial_transformation/spatial_layer1/Tensordot/MatMul:product:0Gmodel/spatial_transformation/spatial_layer1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @╩
Bmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpReadVariableOpKmodel_spatial_transformation_spatial_layer1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0А
3model/spatial_transformation/spatial_layer1/BiasAddBiasAdd>model/spatial_transformation/spatial_layer1/Tensordot:output:0Jmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @и
,model/spatial_transformation/activation/TanhTanh<model/spatial_transformation/spatial_layer1/BiasAdd:output:0*
T0*+
_output_shapes
:         @╥
Dmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOpReadVariableOpMmodel_spatial_transformation_spatial_layer2_tensordot_readvariableop_resource*
_output_shapes

:@@*
dtype0Д
:model/spatial_transformation/spatial_layer2/Tensordot/axesConst*
_output_shapes
:*
dtype0*
valueB:Л
:model/spatial_transformation/spatial_layer2/Tensordot/freeConst*
_output_shapes
:*
dtype0*
valueB"       й
;model/spatial_transformation/spatial_layer2/Tensordot/ShapeShape0model/spatial_transformation/activation/Tanh:y:0*
T0*
_output_shapes
::э╧Е
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
:З
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
:Е
;model/spatial_transformation/spatial_layer2/Tensordot/ConstConst*
_output_shapes
:*
dtype0*
valueB: Є
:model/spatial_transformation/spatial_layer2/Tensordot/ProdProdGmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2:output:0Dmodel/spatial_transformation/spatial_layer2/Tensordot/Const:output:0*
T0*
_output_shapes
: З
=model/spatial_transformation/spatial_layer2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: °
<model/spatial_transformation/spatial_layer2/Tensordot/Prod_1ProdImodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2_1:output:0Fmodel/spatial_transformation/spatial_layer2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: Г
Amodel/spatial_transformation/spatial_layer2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╠
<model/spatial_transformation/spatial_layer2/Tensordot/concatConcatV2Cmodel/spatial_transformation/spatial_layer2/Tensordot/free:output:0Cmodel/spatial_transformation/spatial_layer2/Tensordot/axes:output:0Jmodel/spatial_transformation/spatial_layer2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:¤
;model/spatial_transformation/spatial_layer2/Tensordot/stackPackCmodel/spatial_transformation/spatial_layer2/Tensordot/Prod:output:0Emodel/spatial_transformation/spatial_layer2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:√
?model/spatial_transformation/spatial_layer2/Tensordot/transpose	Transpose0model/spatial_transformation/activation/Tanh:y:0Emodel/spatial_transformation/spatial_layer2/Tensordot/concat:output:0*
T0*+
_output_shapes
:         @О
=model/spatial_transformation/spatial_layer2/Tensordot/ReshapeReshapeCmodel/spatial_transformation/spatial_layer2/Tensordot/transpose:y:0Dmodel/spatial_transformation/spatial_layer2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  О
<model/spatial_transformation/spatial_layer2/Tensordot/MatMulMatMulFmodel/spatial_transformation/spatial_layer2/Tensordot/Reshape:output:0Lmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @З
=model/spatial_transformation/spatial_layer2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Е
Cmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
>model/spatial_transformation/spatial_layer2/Tensordot/concat_1ConcatV2Gmodel/spatial_transformation/spatial_layer2/Tensordot/GatherV2:output:0Fmodel/spatial_transformation/spatial_layer2/Tensordot/Const_2:output:0Lmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:З
5model/spatial_transformation/spatial_layer2/TensordotReshapeFmodel/spatial_transformation/spatial_layer2/Tensordot/MatMul:product:0Gmodel/spatial_transformation/spatial_layer2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @╩
Bmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpReadVariableOpKmodel_spatial_transformation_spatial_layer2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0А
3model/spatial_transformation/spatial_layer2/BiasAddBiasAdd>model/spatial_transformation/spatial_layer2/Tensordot:output:0Jmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @к
.model/spatial_transformation/activation_1/TanhTanh<model/spatial_transformation/spatial_layer2/BiasAdd:output:0*
T0*+
_output_shapes
:         @╚
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0У
.model/multi_head_attention/query/einsum/EinsumEinsum2model/spatial_transformation/activation_1/Tanh:y:0Emodel/multi_head_attention/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abde░
3model/multi_head_attention/query/add/ReadVariableOpReadVariableOp<model_multi_head_attention_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0с
(model/multi_head_attention/query/add/addAddV27model/multi_head_attention/query/einsum/Einsum:output:0;model/multi_head_attention/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @─
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOpReadVariableOpDmodel_multi_head_attention_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Е
,model/multi_head_attention/key/einsum/EinsumEinsummodel/BPE/activation_3/Tanh:y:0Cmodel/multi_head_attention/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdeм
1model/multi_head_attention/key/add/ReadVariableOpReadVariableOp:model_multi_head_attention_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0ф
&model/multi_head_attention/key/add/addAddV25model/multi_head_attention/key/einsum/Einsum:output:09model/multi_head_attention/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @╚
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Й
.model/multi_head_attention/value/einsum/EinsumEinsummodel/BVE/activation_5/Tanh:y:0Emodel/multi_head_attention/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abde░
3model/multi_head_attention/value/add/ReadVariableOpReadVariableOp<model_multi_head_attention_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0ъ
(model/multi_head_attention/value/add/addAddV27model/multi_head_attention/value/einsum/Einsum:output:0;model/multi_head_attention/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @e
 model/multi_head_attention/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >╕
model/multi_head_attention/MulMul,model/multi_head_attention/query/add/add:z:0)model/multi_head_attention/Mul/y:output:0*
T0*/
_output_shapes
:         @э
(model/multi_head_attention/einsum/EinsumEinsum*model/multi_head_attention/key/add/add:z:0"model/multi_head_attention/Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeл
*model/multi_head_attention/softmax/SoftmaxSoftmax1model/multi_head_attention/einsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  ░
+model/multi_head_attention/dropout/IdentityIdentity4model/multi_head_attention/softmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  ·
*model/multi_head_attention/einsum_1/EinsumEinsum4model/multi_head_attention/dropout/Identity:output:0,model/multi_head_attention/value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcd▐
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpQmodel_multi_head_attention_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0ж
9model/multi_head_attention/attention_output/einsum/EinsumEinsum3model/multi_head_attention/einsum_1/Einsum:output:0Pmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abe┬
>model/multi_head_attention/attention_output/add/ReadVariableOpReadVariableOpGmodel_multi_head_attention_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0■
3model/multi_head_attention/attention_output/add/addAddV2Bmodel/multi_head_attention/attention_output/einsum/Einsum:output:0Fmodel/multi_head_attention/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @╣
model/add/addAddV22model/spatial_transformation/activation_1/Tanh:y:07model/multi_head_attention/attention_output/add/add:z:0*
T0*+
_output_shapes
:         @Т
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
::э╧e
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
valueB: Т
model/dense/Tensordot/ProdProd'model/dense/Tensordot/GatherV2:output:0$model/dense/Tensordot/Const:output:0*
T0*
_output_shapes
: g
model/dense/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ш
model/dense/Tensordot/Prod_1Prod)model/dense/Tensordot/GatherV2_1:output:0&model/dense/Tensordot/Const_1:output:0*
T0*
_output_shapes
: c
!model/dense/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╠
model/dense/Tensordot/concatConcatV2#model/dense/Tensordot/free:output:0#model/dense/Tensordot/axes:output:0*model/dense/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:Э
model/dense/Tensordot/stackPack#model/dense/Tensordot/Prod:output:0%model/dense/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:Ь
model/dense/Tensordot/transpose	Transposemodel/add/add:z:0%model/dense/Tensordot/concat:output:0*
T0*+
_output_shapes
:         @о
model/dense/Tensordot/ReshapeReshape#model/dense/Tensordot/transpose:y:0$model/dense/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  о
model/dense/Tensordot/MatMulMatMul&model/dense/Tensordot/Reshape:output:0,model/dense/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @g
model/dense/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@e
#model/dense/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ╫
model/dense/Tensordot/concat_1ConcatV2'model/dense/Tensordot/GatherV2:output:0&model/dense/Tensordot/Const_2:output:0,model/dense/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:з
model/dense/TensordotReshape&model/dense/Tensordot/MatMul:product:0'model/dense/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @К
"model/dense/BiasAdd/ReadVariableOpReadVariableOp+model_dense_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0а
model/dense/BiasAddBiasAddmodel/dense/Tensordot:output:0*model/dense/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @l
model/dense/TanhTanhmodel/dense/BiasAdd:output:0*
T0*+
_output_shapes
:         @Ц
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
valueB"       o
model/dense_1/Tensordot/ShapeShapemodel/dense/Tanh:y:0*
T0*
_output_shapes
::э╧g
%model/dense_1/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : є
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
value	B : ў
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
valueB: Ш
model/dense_1/Tensordot/ProdProd)model/dense_1/Tensordot/GatherV2:output:0&model/dense_1/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_1/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ю
model/dense_1/Tensordot/Prod_1Prod+model/dense_1/Tensordot/GatherV2_1:output:0(model/dense_1/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_1/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╘
model/dense_1/Tensordot/concatConcatV2%model/dense_1/Tensordot/free:output:0%model/dense_1/Tensordot/axes:output:0,model/dense_1/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:г
model/dense_1/Tensordot/stackPack%model/dense_1/Tensordot/Prod:output:0'model/dense_1/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:г
!model/dense_1/Tensordot/transpose	Transposemodel/dense/Tanh:y:0'model/dense_1/Tensordot/concat:output:0*
T0*+
_output_shapes
:         @┤
model/dense_1/Tensordot/ReshapeReshape%model/dense_1/Tensordot/transpose:y:0&model/dense_1/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ┤
model/dense_1/Tensordot/MatMulMatMul(model/dense_1/Tensordot/Reshape:output:0.model/dense_1/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @i
model/dense_1/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_1/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
 model/dense_1/Tensordot/concat_1ConcatV2)model/dense_1/Tensordot/GatherV2:output:0(model/dense_1/Tensordot/Const_2:output:0.model/dense_1/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:н
model/dense_1/TensordotReshape(model/dense_1/Tensordot/MatMul:product:0)model/dense_1/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @О
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
model/dense_1/BiasAddBiasAdd model/dense_1/Tensordot:output:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @p
model/dense_1/TanhTanhmodel/dense_1/BiasAdd:output:0*
T0*+
_output_shapes
:         @╠
?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_1_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0√
0model/multi_head_attention_1/query/einsum/EinsumEinsummodel/dense_1/Tanh:y:0Gmodel/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abde┤
5model/multi_head_attention_1/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_1_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0ч
*model/multi_head_attention_1/query/add/addAddV29model/multi_head_attention_1/query/einsum/Einsum:output:0=model/multi_head_attention_1/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @╚
=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_1_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Й
.model/multi_head_attention_1/key/einsum/EinsumEinsummodel/BPE/activation_3/Tanh:y:0Emodel/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abde░
3model/multi_head_attention_1/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_1_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0ъ
(model/multi_head_attention_1/key/add/addAddV27model/multi_head_attention_1/key/einsum/Einsum:output:0;model/multi_head_attention_1/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @╠
?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_1_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Н
0model/multi_head_attention_1/value/einsum/EinsumEinsummodel/BVE/activation_5/Tanh:y:0Gmodel/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abde┤
5model/multi_head_attention_1/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_1_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Ё
*model/multi_head_attention_1/value/add/addAddV29model/multi_head_attention_1/value/einsum/Einsum:output:0=model/multi_head_attention_1/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @g
"model/multi_head_attention_1/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >╛
 model/multi_head_attention_1/MulMul.model/multi_head_attention_1/query/add/add:z:0+model/multi_head_attention_1/Mul/y:output:0*
T0*/
_output_shapes
:         @є
*model/multi_head_attention_1/einsum/EinsumEinsum,model/multi_head_attention_1/key/add/add:z:0$model/multi_head_attention_1/Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeп
,model/multi_head_attention_1/softmax/SoftmaxSoftmax3model/multi_head_attention_1/einsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  ┤
-model/multi_head_attention_1/dropout/IdentityIdentity6model/multi_head_attention_1/softmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  А
,model/multi_head_attention_1/einsum_1/EinsumEinsum6model/multi_head_attention_1/dropout/Identity:output:0.model/multi_head_attention_1/value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdт
Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_1_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0м
;model/multi_head_attention_1/attention_output/einsum/EinsumEinsum5model/multi_head_attention_1/einsum_1/Einsum:output:0Rmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abe╞
@model/multi_head_attention_1/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_1_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0Д
5model/multi_head_attention_1/attention_output/add/addAddV2Dmodel/multi_head_attention_1/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_1/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @б
model/add_1/addAddV2model/dense_1/Tanh:y:09model/multi_head_attention_1/attention_output/add/add:z:0*
T0*+
_output_shapes
:         @Ц
&model/dense_2/Tensordot/ReadVariableOpReadVariableOp/model_dense_2_tensordot_readvariableop_resource*
_output_shapes

:@@*
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
valueB"       n
model/dense_2/Tensordot/ShapeShapemodel/add_1/add:z:0*
T0*
_output_shapes
::э╧g
%model/dense_2/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : є
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
value	B : ў
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
valueB: Ш
model/dense_2/Tensordot/ProdProd)model/dense_2/Tensordot/GatherV2:output:0&model/dense_2/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_2/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ю
model/dense_2/Tensordot/Prod_1Prod+model/dense_2/Tensordot/GatherV2_1:output:0(model/dense_2/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_2/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╘
model/dense_2/Tensordot/concatConcatV2%model/dense_2/Tensordot/free:output:0%model/dense_2/Tensordot/axes:output:0,model/dense_2/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:г
model/dense_2/Tensordot/stackPack%model/dense_2/Tensordot/Prod:output:0'model/dense_2/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:в
!model/dense_2/Tensordot/transpose	Transposemodel/add_1/add:z:0'model/dense_2/Tensordot/concat:output:0*
T0*+
_output_shapes
:         @┤
model/dense_2/Tensordot/ReshapeReshape%model/dense_2/Tensordot/transpose:y:0&model/dense_2/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ┤
model/dense_2/Tensordot/MatMulMatMul(model/dense_2/Tensordot/Reshape:output:0.model/dense_2/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @i
model/dense_2/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_2/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
 model/dense_2/Tensordot/concat_1ConcatV2)model/dense_2/Tensordot/GatherV2:output:0(model/dense_2/Tensordot/Const_2:output:0.model/dense_2/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:н
model/dense_2/TensordotReshape(model/dense_2/Tensordot/MatMul:product:0)model/dense_2/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @О
$model/dense_2/BiasAdd/ReadVariableOpReadVariableOp-model_dense_2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
model/dense_2/BiasAddBiasAdd model/dense_2/Tensordot:output:0,model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @p
model/dense_2/TanhTanhmodel/dense_2/BiasAdd:output:0*
T0*+
_output_shapes
:         @Ц
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
valueB"       q
model/dense_3/Tensordot/ShapeShapemodel/dense_2/Tanh:y:0*
T0*
_output_shapes
::э╧g
%model/dense_3/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : є
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
value	B : ў
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
valueB: Ш
model/dense_3/Tensordot/ProdProd)model/dense_3/Tensordot/GatherV2:output:0&model/dense_3/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_3/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ю
model/dense_3/Tensordot/Prod_1Prod+model/dense_3/Tensordot/GatherV2_1:output:0(model/dense_3/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_3/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╘
model/dense_3/Tensordot/concatConcatV2%model/dense_3/Tensordot/free:output:0%model/dense_3/Tensordot/axes:output:0,model/dense_3/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:г
model/dense_3/Tensordot/stackPack%model/dense_3/Tensordot/Prod:output:0'model/dense_3/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:е
!model/dense_3/Tensordot/transpose	Transposemodel/dense_2/Tanh:y:0'model/dense_3/Tensordot/concat:output:0*
T0*+
_output_shapes
:         @┤
model/dense_3/Tensordot/ReshapeReshape%model/dense_3/Tensordot/transpose:y:0&model/dense_3/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ┤
model/dense_3/Tensordot/MatMulMatMul(model/dense_3/Tensordot/Reshape:output:0.model/dense_3/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @i
model/dense_3/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_3/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
 model/dense_3/Tensordot/concat_1ConcatV2)model/dense_3/Tensordot/GatherV2:output:0(model/dense_3/Tensordot/Const_2:output:0.model/dense_3/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:н
model/dense_3/TensordotReshape(model/dense_3/Tensordot/MatMul:product:0)model/dense_3/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @О
$model/dense_3/BiasAdd/ReadVariableOpReadVariableOp-model_dense_3_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
model/dense_3/BiasAddBiasAdd model/dense_3/Tensordot:output:0,model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @p
model/dense_3/TanhTanhmodel/dense_3/BiasAdd:output:0*
T0*+
_output_shapes
:         @╠
?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_2_query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0√
0model/multi_head_attention_2/query/einsum/EinsumEinsummodel/dense_3/Tanh:y:0Gmodel/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abde┤
5model/multi_head_attention_2/query/add/ReadVariableOpReadVariableOp>model_multi_head_attention_2_query_add_readvariableop_resource*
_output_shapes

:@*
dtype0ч
*model/multi_head_attention_2/query/add/addAddV29model/multi_head_attention_2/query/einsum/Einsum:output:0=model/multi_head_attention_2/query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @╚
=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOpReadVariableOpFmodel_multi_head_attention_2_key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Й
.model/multi_head_attention_2/key/einsum/EinsumEinsummodel/BPE/activation_3/Tanh:y:0Emodel/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abde░
3model/multi_head_attention_2/key/add/ReadVariableOpReadVariableOp<model_multi_head_attention_2_key_add_readvariableop_resource*
_output_shapes

:@*
dtype0ъ
(model/multi_head_attention_2/key/add/addAddV27model/multi_head_attention_2/key/einsum/Einsum:output:0;model/multi_head_attention_2/key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @╠
?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOpReadVariableOpHmodel_multi_head_attention_2_value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0Н
0model/multi_head_attention_2/value/einsum/EinsumEinsummodel/BVE/activation_5/Tanh:y:0Gmodel/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abde┤
5model/multi_head_attention_2/value/add/ReadVariableOpReadVariableOp>model_multi_head_attention_2_value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Ё
*model/multi_head_attention_2/value/add/addAddV29model/multi_head_attention_2/value/einsum/Einsum:output:0=model/multi_head_attention_2/value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @g
"model/multi_head_attention_2/Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >╛
 model/multi_head_attention_2/MulMul.model/multi_head_attention_2/query/add/add:z:0+model/multi_head_attention_2/Mul/y:output:0*
T0*/
_output_shapes
:         @є
*model/multi_head_attention_2/einsum/EinsumEinsum,model/multi_head_attention_2/key/add/add:z:0$model/multi_head_attention_2/Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeп
,model/multi_head_attention_2/softmax/SoftmaxSoftmax3model/multi_head_attention_2/einsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  ┤
-model/multi_head_attention_2/dropout/IdentityIdentity6model/multi_head_attention_2/softmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  А
,model/multi_head_attention_2/einsum_1/EinsumEinsum6model/multi_head_attention_2/dropout/Identity:output:0.model/multi_head_attention_2/value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdт
Jmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpReadVariableOpSmodel_multi_head_attention_2_attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0м
;model/multi_head_attention_2/attention_output/einsum/EinsumEinsum5model/multi_head_attention_2/einsum_1/Einsum:output:0Rmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abe╞
@model/multi_head_attention_2/attention_output/add/ReadVariableOpReadVariableOpImodel_multi_head_attention_2_attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0Д
5model/multi_head_attention_2/attention_output/add/addAddV2Dmodel/multi_head_attention_2/attention_output/einsum/Einsum:output:0Hmodel/multi_head_attention_2/attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @б
model/add_2/addAddV2model/dense_3/Tanh:y:09model/multi_head_attention_2/attention_output/add/add:z:0*
T0*+
_output_shapes
:         @Ц
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
valueB"       n
model/dense_4/Tensordot/ShapeShapemodel/add_2/add:z:0*
T0*
_output_shapes
::э╧g
%model/dense_4/Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : є
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
value	B : ў
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
valueB: Ш
model/dense_4/Tensordot/ProdProd)model/dense_4/Tensordot/GatherV2:output:0&model/dense_4/Tensordot/Const:output:0*
T0*
_output_shapes
: i
model/dense_4/Tensordot/Const_1Const*
_output_shapes
:*
dtype0*
valueB: Ю
model/dense_4/Tensordot/Prod_1Prod+model/dense_4/Tensordot/GatherV2_1:output:0(model/dense_4/Tensordot/Const_1:output:0*
T0*
_output_shapes
: e
#model/dense_4/Tensordot/concat/axisConst*
_output_shapes
: *
dtype0*
value	B : ╘
model/dense_4/Tensordot/concatConcatV2%model/dense_4/Tensordot/free:output:0%model/dense_4/Tensordot/axes:output:0,model/dense_4/Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:г
model/dense_4/Tensordot/stackPack%model/dense_4/Tensordot/Prod:output:0'model/dense_4/Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:в
!model/dense_4/Tensordot/transpose	Transposemodel/add_2/add:z:0'model/dense_4/Tensordot/concat:output:0*
T0*+
_output_shapes
:         @┤
model/dense_4/Tensordot/ReshapeReshape%model/dense_4/Tensordot/transpose:y:0&model/dense_4/Tensordot/stack:output:0*
T0*0
_output_shapes
:                  ┤
model/dense_4/Tensordot/MatMulMatMul(model/dense_4/Tensordot/Reshape:output:0.model/dense_4/Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @i
model/dense_4/Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@g
%model/dense_4/Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : ▀
 model/dense_4/Tensordot/concat_1ConcatV2)model/dense_4/Tensordot/GatherV2:output:0(model/dense_4/Tensordot/Const_2:output:0.model/dense_4/Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:н
model/dense_4/TensordotReshape(model/dense_4/Tensordot/MatMul:product:0)model/dense_4/Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @О
$model/dense_4/BiasAdd/ReadVariableOpReadVariableOp-model_dense_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0ж
model/dense_4/BiasAddBiasAdd model/dense_4/Tensordot:output:0,model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @p
model/dense_4/TanhTanhmodel/dense_4/BiasAdd:output:0*
T0*+
_output_shapes
:         @d
model/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   И
model/flatten/ReshapeReshapemodel/dense_4/Tanh:y:0model/flatten/Const:output:0*
T0*'
_output_shapes
:         @М
!model/U/ou1/MatMul/ReadVariableOpReadVariableOp*model_u_ou1_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Щ
model/U/ou1/MatMulMatMulmodel/flatten/Reshape:output:0)model/U/ou1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @К
"model/U/ou1/BiasAdd/ReadVariableOpReadVariableOp+model_u_ou1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
model/U/ou1/BiasAddBiasAddmodel/U/ou1/MatMul:product:0*model/U/ou1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @q
model/U/activation_6/TanhTanhmodel/U/ou1/BiasAdd:output:0*
T0*'
_output_shapes
:         @М
!model/U/ou2/MatMul/ReadVariableOpReadVariableOp*model_u_ou2_matmul_readvariableop_resource*
_output_shapes

:@@*
dtype0Ш
model/U/ou2/MatMulMatMulmodel/U/activation_6/Tanh:y:0)model/U/ou2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @К
"model/U/ou2/BiasAdd/ReadVariableOpReadVariableOp+model_u_ou2_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Ъ
model/U/ou2/BiasAddBiasAddmodel/U/ou2/MatMul:product:0*model/U/ou2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @q
model/U/activation_7/TanhTanhmodel/U/ou2/BiasAdd:output:0*
T0*'
_output_shapes
:         @Т
$model/output_u/MatMul/ReadVariableOpReadVariableOp-model_output_u_matmul_readvariableop_resource*
_output_shapes

:@*
dtype0Ю
model/output_u/MatMulMatMulmodel/U/activation_7/Tanh:y:0,model/output_u/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         Р
%model/output_u/BiasAdd/ReadVariableOpReadVariableOp.model_output_u_biasadd_readvariableop_resource*
_output_shapes
:*
dtype0г
model/output_u/BiasAddBiasAddmodel/output_u/MatMul:product:0-model/output_u/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         n
IdentityIdentitymodel/output_u/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         ¤
NoOpNoOp,^model/BPE/bpe_layer1/BiasAdd/ReadVariableOp.^model/BPE/bpe_layer1/Tensordot/ReadVariableOp,^model/BPE/bpe_layer2/BiasAdd/ReadVariableOp.^model/BPE/bpe_layer2/Tensordot/ReadVariableOp,^model/BVE/bve_layer1/BiasAdd/ReadVariableOp.^model/BVE/bve_layer1/Tensordot/ReadVariableOp,^model/BVE/bve_layer2/BiasAdd/ReadVariableOp.^model/BVE/bve_layer2/Tensordot/ReadVariableOp#^model/U/ou1/BiasAdd/ReadVariableOp"^model/U/ou1/MatMul/ReadVariableOp#^model/U/ou2/BiasAdd/ReadVariableOp"^model/U/ou2/MatMul/ReadVariableOp#^model/dense/BiasAdd/ReadVariableOp%^model/dense/Tensordot/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp'^model/dense_1/Tensordot/ReadVariableOp%^model/dense_2/BiasAdd/ReadVariableOp'^model/dense_2/Tensordot/ReadVariableOp%^model/dense_3/BiasAdd/ReadVariableOp'^model/dense_3/Tensordot/ReadVariableOp%^model/dense_4/BiasAdd/ReadVariableOp'^model/dense_4/Tensordot/ReadVariableOp?^model/multi_head_attention/attention_output/add/ReadVariableOpI^model/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2^model/multi_head_attention/key/add/ReadVariableOp<^model/multi_head_attention/key/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/query/add/ReadVariableOp>^model/multi_head_attention/query/einsum/Einsum/ReadVariableOp4^model/multi_head_attention/value/add/ReadVariableOp>^model/multi_head_attention/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_1/attention_output/add/ReadVariableOpK^model/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_1/key/add/ReadVariableOp>^model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_1/query/add/ReadVariableOp@^model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_1/value/add/ReadVariableOp@^model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOpA^model/multi_head_attention_2/attention_output/add/ReadVariableOpK^model/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp4^model/multi_head_attention_2/key/add/ReadVariableOp>^model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_2/query/add/ReadVariableOp@^model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp6^model/multi_head_attention_2/value/add/ReadVariableOp@^model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp&^model/output_u/BiasAdd/ReadVariableOp%^model/output_u/MatMul/ReadVariableOpC^model/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpE^model/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpC^model/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpE^model/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesё
ю:         :         :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2Z
+model/BPE/bpe_layer1/BiasAdd/ReadVariableOp+model/BPE/bpe_layer1/BiasAdd/ReadVariableOp2^
-model/BPE/bpe_layer1/Tensordot/ReadVariableOp-model/BPE/bpe_layer1/Tensordot/ReadVariableOp2Z
+model/BPE/bpe_layer2/BiasAdd/ReadVariableOp+model/BPE/bpe_layer2/BiasAdd/ReadVariableOp2^
-model/BPE/bpe_layer2/Tensordot/ReadVariableOp-model/BPE/bpe_layer2/Tensordot/ReadVariableOp2Z
+model/BVE/bve_layer1/BiasAdd/ReadVariableOp+model/BVE/bve_layer1/BiasAdd/ReadVariableOp2^
-model/BVE/bve_layer1/Tensordot/ReadVariableOp-model/BVE/bve_layer1/Tensordot/ReadVariableOp2Z
+model/BVE/bve_layer2/BiasAdd/ReadVariableOp+model/BVE/bve_layer2/BiasAdd/ReadVariableOp2^
-model/BVE/bve_layer2/Tensordot/ReadVariableOp-model/BVE/bve_layer2/Tensordot/ReadVariableOp2H
"model/U/ou1/BiasAdd/ReadVariableOp"model/U/ou1/BiasAdd/ReadVariableOp2F
!model/U/ou1/MatMul/ReadVariableOp!model/U/ou1/MatMul/ReadVariableOp2H
"model/U/ou2/BiasAdd/ReadVariableOp"model/U/ou2/BiasAdd/ReadVariableOp2F
!model/U/ou2/MatMul/ReadVariableOp!model/U/ou2/MatMul/ReadVariableOp2H
"model/dense/BiasAdd/ReadVariableOp"model/dense/BiasAdd/ReadVariableOp2L
$model/dense/Tensordot/ReadVariableOp$model/dense/Tensordot/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2P
&model/dense_1/Tensordot/ReadVariableOp&model/dense_1/Tensordot/ReadVariableOp2L
$model/dense_2/BiasAdd/ReadVariableOp$model/dense_2/BiasAdd/ReadVariableOp2P
&model/dense_2/Tensordot/ReadVariableOp&model/dense_2/Tensordot/ReadVariableOp2L
$model/dense_3/BiasAdd/ReadVariableOp$model/dense_3/BiasAdd/ReadVariableOp2P
&model/dense_3/Tensordot/ReadVariableOp&model/dense_3/Tensordot/ReadVariableOp2L
$model/dense_4/BiasAdd/ReadVariableOp$model/dense_4/BiasAdd/ReadVariableOp2P
&model/dense_4/Tensordot/ReadVariableOp&model/dense_4/Tensordot/ReadVariableOp2А
>model/multi_head_attention/attention_output/add/ReadVariableOp>model/multi_head_attention/attention_output/add/ReadVariableOp2Ф
Hmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOpHmodel/multi_head_attention/attention_output/einsum/Einsum/ReadVariableOp2f
1model/multi_head_attention/key/add/ReadVariableOp1model/multi_head_attention/key/add/ReadVariableOp2z
;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp;model/multi_head_attention/key/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/query/add/ReadVariableOp3model/multi_head_attention/query/add/ReadVariableOp2~
=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp=model/multi_head_attention/query/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention/value/add/ReadVariableOp3model/multi_head_attention/value/add/ReadVariableOp2~
=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp=model/multi_head_attention/value/einsum/Einsum/ReadVariableOp2Д
@model/multi_head_attention_1/attention_output/add/ReadVariableOp@model/multi_head_attention_1/attention_output/add/ReadVariableOp2Ш
Jmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_1/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_1/key/add/ReadVariableOp3model/multi_head_attention_1/key/add/ReadVariableOp2~
=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_1/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_1/query/add/ReadVariableOp5model/multi_head_attention_1/query/add/ReadVariableOp2В
?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_1/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_1/value/add/ReadVariableOp5model/multi_head_attention_1/value/add/ReadVariableOp2В
?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_1/value/einsum/Einsum/ReadVariableOp2Д
@model/multi_head_attention_2/attention_output/add/ReadVariableOp@model/multi_head_attention_2/attention_output/add/ReadVariableOp2Ш
Jmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOpJmodel/multi_head_attention_2/attention_output/einsum/Einsum/ReadVariableOp2j
3model/multi_head_attention_2/key/add/ReadVariableOp3model/multi_head_attention_2/key/add/ReadVariableOp2~
=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp=model/multi_head_attention_2/key/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_2/query/add/ReadVariableOp5model/multi_head_attention_2/query/add/ReadVariableOp2В
?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp?model/multi_head_attention_2/query/einsum/Einsum/ReadVariableOp2n
5model/multi_head_attention_2/value/add/ReadVariableOp5model/multi_head_attention_2/value/add/ReadVariableOp2В
?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp?model/multi_head_attention_2/value/einsum/Einsum/ReadVariableOp2N
%model/output_u/BiasAdd/ReadVariableOp%model/output_u/BiasAdd/ReadVariableOp2L
$model/output_u/MatMul/ReadVariableOp$model/output_u/MatMul/ReadVariableOp2И
Bmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOpBmodel/spatial_transformation/spatial_layer1/BiasAdd/ReadVariableOp2М
Dmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOpDmodel/spatial_transformation/spatial_layer1/Tensordot/ReadVariableOp2И
Bmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOpBmodel/spatial_transformation/spatial_layer2/BiasAdd/ReadVariableOp2М
Dmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOpDmodel/spatial_transformation/spatial_layer2/Tensordot/ReadVariableOp:(8$
"
_user_specified_name
resource:(7$
"
_user_specified_name
resource:(6$
"
_user_specified_name
resource:(5$
"
_user_specified_name
resource:(4$
"
_user_specified_name
resource:(3$
"
_user_specified_name
resource:(2$
"
_user_specified_name
resource:(1$
"
_user_specified_name
resource:(0$
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
 :                  
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:         
!
_user_specified_name	t_input:P L
'
_output_shapes
:         
!
_user_specified_name	x_input
б.
Т
'__inference_model_layer_call_fn_2999267
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layer
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
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

unknown_23:@@

unknown_24:@ 

unknown_25:@@

unknown_26:@ 

unknown_27:@@

unknown_28:@ 

unknown_29:@@

unknown_30:@

unknown_31:@@

unknown_32:@

unknown_33:@@

unknown_34:@ 

unknown_35:@@

unknown_36:@ 

unknown_37:@@

unknown_38:@ 

unknown_39:@@

unknown_40:@ 

unknown_41:@@

unknown_42:@

unknown_43:@@

unknown_44:@

unknown_45:@@

unknown_46:@

unknown_47:@@

unknown_48:@

unknown_49:@

unknown_50:
identityИвStatefulPartitionedCall╝
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
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2999041o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesё
ю:         :         :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'8#
!
_user_specified_name	2999263:'7#
!
_user_specified_name	2999261:'6#
!
_user_specified_name	2999259:'5#
!
_user_specified_name	2999257:'4#
!
_user_specified_name	2999255:'3#
!
_user_specified_name	2999253:'2#
!
_user_specified_name	2999251:'1#
!
_user_specified_name	2999249:'0#
!
_user_specified_name	2999247:'/#
!
_user_specified_name	2999245:'.#
!
_user_specified_name	2999243:'-#
!
_user_specified_name	2999241:',#
!
_user_specified_name	2999239:'+#
!
_user_specified_name	2999237:'*#
!
_user_specified_name	2999235:')#
!
_user_specified_name	2999233:'(#
!
_user_specified_name	2999231:''#
!
_user_specified_name	2999229:'&#
!
_user_specified_name	2999227:'%#
!
_user_specified_name	2999225:'$#
!
_user_specified_name	2999223:'##
!
_user_specified_name	2999221:'"#
!
_user_specified_name	2999219:'!#
!
_user_specified_name	2999217:' #
!
_user_specified_name	2999215:'#
!
_user_specified_name	2999213:'#
!
_user_specified_name	2999211:'#
!
_user_specified_name	2999209:'#
!
_user_specified_name	2999207:'#
!
_user_specified_name	2999205:'#
!
_user_specified_name	2999203:'#
!
_user_specified_name	2999201:'#
!
_user_specified_name	2999199:'#
!
_user_specified_name	2999197:'#
!
_user_specified_name	2999195:'#
!
_user_specified_name	2999193:'#
!
_user_specified_name	2999191:'#
!
_user_specified_name	2999189:'#
!
_user_specified_name	2999187:'#
!
_user_specified_name	2999185:'#
!
_user_specified_name	2999183:'#
!
_user_specified_name	2999181:'#
!
_user_specified_name	2999179:'#
!
_user_specified_name	2999177:'#
!
_user_specified_name	2999175:'#
!
_user_specified_name	2999173:'
#
!
_user_specified_name	2999171:'	#
!
_user_specified_name	2999169:'#
!
_user_specified_name	2999167:'#
!
_user_specified_name	2999165:'#
!
_user_specified_name	2999163:'#
!
_user_specified_name	2999161:_[
4
_output_shapes"
 :                  
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:         
!
_user_specified_name	t_input:P L
'
_output_shapes
:         
!
_user_specified_name	x_input
▄
I
-__inference_rescaling_2_layer_call_fn_2999677

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_2_layer_call_and_return_conditional_losses_2998337m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╞
b
F__inference_rescaling_layer_call_and_return_conditional_losses_2998319

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
 *  А┐U
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:         Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
Р
В
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_2997753

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
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
:         
 
_user_specified_nameinputs
И
Р
>__inference_U_layer_call_and_return_conditional_losses_2998251
	ou1_input
ou1_2998238:@@
ou1_2998240:@
ou2_2998244:@@
ou2_2998246:@
identityИвou1/StatefulPartitionedCallвou2/StatefulPartitionedCallх
ou1/StatefulPartitionedCallStatefulPartitionedCall	ou1_inputou1_2998238ou1_2998240*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_ou1_layer_call_and_return_conditional_losses_2998201т
activation_6/PartitionedCallPartitionedCall$ou1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_2998211Б
ou2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0ou2_2998244ou2_2998246*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_ou2_layer_call_and_return_conditional_losses_2998222т
activation_7/PartitionedCallPartitionedCall$ou2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_2998232t
IdentityIdentity%activation_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @^
NoOpNoOp^ou1/StatefulPartitionedCall^ou2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2:
ou1/StatefulPartitionedCallou1/StatefulPartitionedCall2:
ou2/StatefulPartitionedCallou2/StatefulPartitionedCall:'#
!
_user_specified_name	2998246:'#
!
_user_specified_name	2998244:'#
!
_user_specified_name	2998240:'#
!
_user_specified_name	2998238:R N
'
_output_shapes
:         @
#
_user_specified_name	ou1_input
╞
Ў
8__inference_multi_head_attention_2_layer_call_fn_3000195	
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
identityИвStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_2999001s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	3000191:'	#
!
_user_specified_name	3000189:'#
!
_user_specified_name	3000187:'#
!
_user_specified_name	3000185:'#
!
_user_specified_name	3000183:'#
!
_user_specified_name	3000181:'#
!
_user_specified_name	3000179:'#
!
_user_specified_name	3000177:YU
4
_output_shapes"
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
╞
Ў
8__inference_multi_head_attention_1_layer_call_fn_2999985	
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
identityИвStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_2998937s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	2999981:'	#
!
_user_specified_name	2999979:'#
!
_user_specified_name	2999977:'#
!
_user_specified_name	2999975:'#
!
_user_specified_name	2999973:'#
!
_user_specified_name	2999971:'#
!
_user_specified_name	2999969:'#
!
_user_specified_name	2999967:YU
4
_output_shapes"
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
╞
Ў
8__inference_multi_head_attention_1_layer_call_fn_2999962	
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
identityИвStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_2998572s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	2999958:'	#
!
_user_specified_name	2999956:'#
!
_user_specified_name	2999954:'#
!
_user_specified_name	2999952:'#
!
_user_specified_name	2999950:'#
!
_user_specified_name	2999948:'#
!
_user_specified_name	2999946:'#
!
_user_specified_name	2999944:YU
4
_output_shapes"
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
К
t
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2998362

inputs
inputs_1
identityM
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :В
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*4
_output_shapes"
 :                  d
IdentityIdentityconcat:output:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:                  :                  :\X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
▐
J
.__inference_activation_3_layer_call_fn_3000540

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_2997960m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
─
■
G__inference_bpe_layer2_layer_call_and_return_conditional_losses_3000535

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 20
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
 :                  @
 
_user_specified_nameinputs
Г
Ц
)__inference_dense_2_layer_call_fn_3000078

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_2998627s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000074:'#
!
_user_specified_name	3000072:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
┐.
Е
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_3000021	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
Х
├
@__inference_BPE_layer_call_and_return_conditional_losses_2997979
bpe_layer1_input$
bpe_layer1_2997966:@ 
bpe_layer1_2997968:@$
bpe_layer2_2997972:@@ 
bpe_layer2_2997974:@
identityИв"bpe_layer1/StatefulPartitionedCallв"bpe_layer2/StatefulPartitionedCallХ
"bpe_layer1/StatefulPartitionedCallStatefulPartitionedCallbpe_layer1_inputbpe_layer1_2997966bpe_layer1_2997968*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bpe_layer1_layer_call_and_return_conditional_losses_2997909Ў
activation_2/PartitionedCallPartitionedCall+bpe_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_2997919к
"bpe_layer2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0bpe_layer2_2997972bpe_layer2_2997974*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bpe_layer2_layer_call_and_return_conditional_losses_2997950Ў
activation_3/PartitionedCallPartitionedCall+bpe_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_2997960Б
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @l
NoOpNoOp#^bpe_layer1/StatefulPartitionedCall#^bpe_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 2H
"bpe_layer1/StatefulPartitionedCall"bpe_layer1/StatefulPartitionedCall2H
"bpe_layer2/StatefulPartitionedCall"bpe_layer2/StatefulPartitionedCall:'#
!
_user_specified_name	2997974:'#
!
_user_specified_name	2997972:'#
!
_user_specified_name	2997968:'#
!
_user_specified_name	2997966:f b
4
_output_shapes"
 :                  
*
_user_specified_namebpe_layer1_input
╚
d
H__inference_rescaling_1_layer_call_and_return_conditional_losses_2999659

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
 *  А┐U
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:         Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
 
Ф
'__inference_dense_layer_call_fn_2999868

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCall▐
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2998495s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2999864:'#
!
_user_specified_name	2999862:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ў
e
I__inference_activation_5_layer_call_and_return_conditional_losses_2998116

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :                  @]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Р
В
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_3000437

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
─
■
G__inference_bpe_layer1_layer_call_and_return_conditional_losses_2997909

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
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
 :                  
 
_user_specified_nameinputs
═
j
@__inference_add_layer_call_and_return_conditional_losses_2998463

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:         @S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:SO
+
_output_shapes
:         @
 
_user_specified_nameinputs:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
┴
t
H__inference_concatenate_layer_call_and_return_conditional_losses_2999672
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
:         W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :QM
'
_output_shapes
:         
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs_0
╝
`
D__inference_flatten_layer_call_and_return_conditional_losses_3000330

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         @X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
├
e
I__inference_activation_6_layer_call_and_return_conditional_losses_3000672

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:         @P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ы
Т
%__inference_ou2_layer_call_fn_3000681

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_ou2_layer_call_and_return_conditional_losses_2998222o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000677:'#
!
_user_specified_name	3000675:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
─
■
G__inference_bve_layer1_layer_call_and_return_conditional_losses_2998065

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
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
 :                  
 
_user_specified_nameinputs
▄
I
-__inference_rescaling_3_layer_call_fn_2999690

inputs
identity├
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_3_layer_call_and_return_conditional_losses_2998346m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╫
√
D__inference_dense_2_layer_call_and_return_conditional_losses_3000109

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
╒
l
@__inference_add_layer_call_and_return_conditional_losses_2999859
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:         @S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:UQ
+
_output_shapes
:         @
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:         @
"
_user_specified_name
inputs_0
Г
Ц
)__inference_dense_4_layer_call_fn_3000288

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2998759s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000284:'#
!
_user_specified_name	3000282:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Ц	
╔
#__inference_U_layer_call_fn_2998264
	ou1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCall	ou1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_U_layer_call_and_return_conditional_losses_2998235o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2998260:'#
!
_user_specified_name	2998258:'#
!
_user_specified_name	2998256:'#
!
_user_specified_name	2998254:R N
'
_output_shapes
:         @
#
_user_specified_name	ou1_input
┐.
Е
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_2999001	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
╜.
Г
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2999847	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
н
Щ
,__inference_bve_layer2_layer_call_fn_3000603

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bve_layer2_layer_call_and_return_conditional_losses_2998106|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000599:'#
!
_user_specified_name	3000597:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Ц	
╔
#__inference_U_layer_call_fn_2998277
	ou1_input
unknown:@@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityИвStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCall	ou1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_U_layer_call_and_return_conditional_losses_2998251o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2998273:'#
!
_user_specified_name	2998271:'#
!
_user_specified_name	2998269:'#
!
_user_specified_name	2998267:R N
'
_output_shapes
:         @
#
_user_specified_name	ou1_input
Р
В
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_3000388

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @c
IdentityIdentityBiasAdd:output:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         : : 20
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
:         
 
_user_specified_nameinputs
▐
J
.__inference_activation_5_layer_call_fn_3000638

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_2998116m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╫
√
D__inference_dense_3_layer_call_and_return_conditional_losses_2998663

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
Ує
Ъ#
#__inference__traced_restore_3001204
file_prefix/
assignvariableop_dense_kernel:@@+
assignvariableop_1_dense_bias:@3
!assignvariableop_2_dense_1_kernel:@@-
assignvariableop_3_dense_1_bias:@3
!assignvariableop_4_dense_2_kernel:@@-
assignvariableop_5_dense_2_bias:@3
!assignvariableop_6_dense_3_kernel:@@-
assignvariableop_7_dense_3_bias:@3
!assignvariableop_8_dense_4_kernel:@@-
assignvariableop_9_dense_4_bias:@5
#assignvariableop_10_output_u_kernel:@/
!assignvariableop_11_output_u_bias:;
)assignvariableop_12_spatial_layer1_kernel:@5
'assignvariableop_13_spatial_layer1_bias:@;
)assignvariableop_14_spatial_layer2_kernel:@@5
'assignvariableop_15_spatial_layer2_bias:@7
%assignvariableop_16_bpe_layer1_kernel:@1
#assignvariableop_17_bpe_layer1_bias:@7
%assignvariableop_18_bpe_layer2_kernel:@@1
#assignvariableop_19_bpe_layer2_bias:@7
%assignvariableop_20_bve_layer1_kernel:@1
#assignvariableop_21_bve_layer1_bias:@7
%assignvariableop_22_bve_layer2_kernel:@@1
#assignvariableop_23_bve_layer2_bias:@K
5assignvariableop_24_multi_head_attention_query_kernel:@@E
3assignvariableop_25_multi_head_attention_query_bias:@I
3assignvariableop_26_multi_head_attention_key_kernel:@@C
1assignvariableop_27_multi_head_attention_key_bias:@K
5assignvariableop_28_multi_head_attention_value_kernel:@@E
3assignvariableop_29_multi_head_attention_value_bias:@V
@assignvariableop_30_multi_head_attention_attention_output_kernel:@@L
>assignvariableop_31_multi_head_attention_attention_output_bias:@M
7assignvariableop_32_multi_head_attention_1_query_kernel:@@G
5assignvariableop_33_multi_head_attention_1_query_bias:@K
5assignvariableop_34_multi_head_attention_1_key_kernel:@@E
3assignvariableop_35_multi_head_attention_1_key_bias:@M
7assignvariableop_36_multi_head_attention_1_value_kernel:@@G
5assignvariableop_37_multi_head_attention_1_value_bias:@X
Bassignvariableop_38_multi_head_attention_1_attention_output_kernel:@@N
@assignvariableop_39_multi_head_attention_1_attention_output_bias:@M
7assignvariableop_40_multi_head_attention_2_query_kernel:@@G
5assignvariableop_41_multi_head_attention_2_query_bias:@K
5assignvariableop_42_multi_head_attention_2_key_kernel:@@E
3assignvariableop_43_multi_head_attention_2_key_bias:@M
7assignvariableop_44_multi_head_attention_2_value_kernel:@@G
5assignvariableop_45_multi_head_attention_2_value_bias:@X
Bassignvariableop_46_multi_head_attention_2_attention_output_kernel:@@N
@assignvariableop_47_multi_head_attention_2_attention_output_bias:@0
assignvariableop_48_ou1_kernel:@@*
assignvariableop_49_ou1_bias:@0
assignvariableop_50_ou2_kernel:@@*
assignvariableop_51_ou2_bias:@
identity_53ИвAssignVariableOpвAssignVariableOp_1вAssignVariableOp_10вAssignVariableOp_11вAssignVariableOp_12вAssignVariableOp_13вAssignVariableOp_14вAssignVariableOp_15вAssignVariableOp_16вAssignVariableOp_17вAssignVariableOp_18вAssignVariableOp_19вAssignVariableOp_2вAssignVariableOp_20вAssignVariableOp_21вAssignVariableOp_22вAssignVariableOp_23вAssignVariableOp_24вAssignVariableOp_25вAssignVariableOp_26вAssignVariableOp_27вAssignVariableOp_28вAssignVariableOp_29вAssignVariableOp_3вAssignVariableOp_30вAssignVariableOp_31вAssignVariableOp_32вAssignVariableOp_33вAssignVariableOp_34вAssignVariableOp_35вAssignVariableOp_36вAssignVariableOp_37вAssignVariableOp_38вAssignVariableOp_39вAssignVariableOp_4вAssignVariableOp_40вAssignVariableOp_41вAssignVariableOp_42вAssignVariableOp_43вAssignVariableOp_44вAssignVariableOp_45вAssignVariableOp_46вAssignVariableOp_47вAssignVariableOp_48вAssignVariableOp_49вAssignVariableOp_5вAssignVariableOp_50вAssignVariableOp_51вAssignVariableOp_6вAssignVariableOp_7вAssignVariableOp_8вAssignVariableOp_9Г
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*й
valueЯBЬ5B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH┌
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B к
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*ъ
_output_shapes╫
╘:::::::::::::::::::::::::::::::::::::::::::::::::::::*C
dtypes9
725[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:░
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:┤
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_4AssignVariableOp!assignvariableop_4_dense_2_kernelIdentity_4:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_5AssignVariableOpassignvariableop_5_dense_2_biasIdentity_5:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_3_kernelIdentity_6:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_3_biasIdentity_7:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:╕
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_4_kernelIdentity_8:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:╢
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_4_biasIdentity_9:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_10AssignVariableOp#assignvariableop_10_output_u_kernelIdentity_10:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:║
AssignVariableOp_11AssignVariableOp!assignvariableop_11_output_u_biasIdentity_11:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_12AssignVariableOp)assignvariableop_12_spatial_layer1_kernelIdentity_12:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_13AssignVariableOp'assignvariableop_13_spatial_layer1_biasIdentity_13:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:┬
AssignVariableOp_14AssignVariableOp)assignvariableop_14_spatial_layer2_kernelIdentity_14:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:└
AssignVariableOp_15AssignVariableOp'assignvariableop_15_spatial_layer2_biasIdentity_15:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_16AssignVariableOp%assignvariableop_16_bpe_layer1_kernelIdentity_16:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_17AssignVariableOp#assignvariableop_17_bpe_layer1_biasIdentity_17:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_18AssignVariableOp%assignvariableop_18_bpe_layer2_kernelIdentity_18:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_19AssignVariableOp#assignvariableop_19_bpe_layer2_biasIdentity_19:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_20AssignVariableOp%assignvariableop_20_bve_layer1_kernelIdentity_20:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_21AssignVariableOp#assignvariableop_21_bve_layer1_biasIdentity_21:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:╛
AssignVariableOp_22AssignVariableOp%assignvariableop_22_bve_layer2_kernelIdentity_22:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:╝
AssignVariableOp_23AssignVariableOp#assignvariableop_23_bve_layer2_biasIdentity_23:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_24AssignVariableOp5assignvariableop_24_multi_head_attention_query_kernelIdentity_24:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_25AssignVariableOp3assignvariableop_25_multi_head_attention_query_biasIdentity_25:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_26AssignVariableOp3assignvariableop_26_multi_head_attention_key_kernelIdentity_26:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:╩
AssignVariableOp_27AssignVariableOp1assignvariableop_27_multi_head_attention_key_biasIdentity_27:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_28AssignVariableOp5assignvariableop_28_multi_head_attention_value_kernelIdentity_28:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_29AssignVariableOp3assignvariableop_29_multi_head_attention_value_biasIdentity_29:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_30AssignVariableOp@assignvariableop_30_multi_head_attention_attention_output_kernelIdentity_30:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:╫
AssignVariableOp_31AssignVariableOp>assignvariableop_31_multi_head_attention_attention_output_biasIdentity_31:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:╨
AssignVariableOp_32AssignVariableOp7assignvariableop_32_multi_head_attention_1_query_kernelIdentity_32:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_33AssignVariableOp5assignvariableop_33_multi_head_attention_1_query_biasIdentity_33:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_34AssignVariableOp5assignvariableop_34_multi_head_attention_1_key_kernelIdentity_34:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_35AssignVariableOp3assignvariableop_35_multi_head_attention_1_key_biasIdentity_35:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:╨
AssignVariableOp_36AssignVariableOp7assignvariableop_36_multi_head_attention_1_value_kernelIdentity_36:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_37AssignVariableOp5assignvariableop_37_multi_head_attention_1_value_biasIdentity_37:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:█
AssignVariableOp_38AssignVariableOpBassignvariableop_38_multi_head_attention_1_attention_output_kernelIdentity_38:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_39AssignVariableOp@assignvariableop_39_multi_head_attention_1_attention_output_biasIdentity_39:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:╨
AssignVariableOp_40AssignVariableOp7assignvariableop_40_multi_head_attention_2_query_kernelIdentity_40:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_41AssignVariableOp5assignvariableop_41_multi_head_attention_2_query_biasIdentity_41:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_42AssignVariableOp5assignvariableop_42_multi_head_attention_2_key_kernelIdentity_42:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:╠
AssignVariableOp_43AssignVariableOp3assignvariableop_43_multi_head_attention_2_key_biasIdentity_43:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:╨
AssignVariableOp_44AssignVariableOp7assignvariableop_44_multi_head_attention_2_value_kernelIdentity_44:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:╬
AssignVariableOp_45AssignVariableOp5assignvariableop_45_multi_head_attention_2_value_biasIdentity_45:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:█
AssignVariableOp_46AssignVariableOpBassignvariableop_46_multi_head_attention_2_attention_output_kernelIdentity_46:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:┘
AssignVariableOp_47AssignVariableOp@assignvariableop_47_multi_head_attention_2_attention_output_biasIdentity_47:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_48AssignVariableOpassignvariableop_48_ou1_kernelIdentity_48:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_49AssignVariableOpassignvariableop_49_ou1_biasIdentity_49:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:╖
AssignVariableOp_50AssignVariableOpassignvariableop_50_ou2_kernelIdentity_50:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0_
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:╡
AssignVariableOp_51AssignVariableOpassignvariableop_51_ou2_biasIdentity_51:output:0"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *
dtype0Y
NoOpNoOp"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 ╟	
Identity_52Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_53IdentityIdentity_52:output:0^NoOp_1*
T0*
_output_shapes
: Р	
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
_output_shapes
 "#
identity_53Identity_53:output:0*(
_construction_contextkEagerRuntime*}
_input_shapesl
j: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2*
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
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_92$
AssignVariableOpAssignVariableOp:(4$
"
_user_specified_name
ou2/bias:*3&
$
_user_specified_name
ou2/kernel:(2$
"
_user_specified_name
ou1/bias:*1&
$
_user_specified_name
ou1/kernel:L0H
F
_user_specified_name.,multi_head_attention_2/attention_output/bias:N/J
H
_user_specified_name0.multi_head_attention_2/attention_output/kernel:A.=
;
_user_specified_name#!multi_head_attention_2/value/bias:C-?
=
_user_specified_name%#multi_head_attention_2/value/kernel:?,;
9
_user_specified_name!multi_head_attention_2/key/bias:A+=
;
_user_specified_name#!multi_head_attention_2/key/kernel:A*=
;
_user_specified_name#!multi_head_attention_2/query/bias:C)?
=
_user_specified_name%#multi_head_attention_2/query/kernel:L(H
F
_user_specified_name.,multi_head_attention_1/attention_output/bias:N'J
H
_user_specified_name0.multi_head_attention_1/attention_output/kernel:A&=
;
_user_specified_name#!multi_head_attention_1/value/bias:C%?
=
_user_specified_name%#multi_head_attention_1/value/kernel:?$;
9
_user_specified_name!multi_head_attention_1/key/bias:A#=
;
_user_specified_name#!multi_head_attention_1/key/kernel:A"=
;
_user_specified_name#!multi_head_attention_1/query/bias:C!?
=
_user_specified_name%#multi_head_attention_1/query/kernel:J F
D
_user_specified_name,*multi_head_attention/attention_output/bias:LH
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?;
9
_user_specified_name!multi_head_attention/value/bias:A=
;
_user_specified_name#!multi_head_attention/value/kernel:=9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:/+
)
_user_specified_namebve_layer2/bias:1-
+
_user_specified_namebve_layer2/kernel:/+
)
_user_specified_namebve_layer1/bias:1-
+
_user_specified_namebve_layer1/kernel:/+
)
_user_specified_namebpe_layer2/bias:1-
+
_user_specified_namebpe_layer2/kernel:/+
)
_user_specified_namebpe_layer1/bias:1-
+
_user_specified_namebpe_layer1/kernel:3/
-
_user_specified_namespatial_layer2/bias:51
/
_user_specified_namespatial_layer2/kernel:3/
-
_user_specified_namespatial_layer1/bias:51
/
_user_specified_namespatial_layer1/kernel:-)
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
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
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
─
■
G__inference_bpe_layer1_layer_call_and_return_conditional_losses_3000486

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
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
 :                  
 
_user_specified_nameinputs
н
Щ
,__inference_bpe_layer2_layer_call_fn_3000505

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bpe_layer2_layer_call_and_return_conditional_losses_2997950|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000501:'#
!
_user_specified_name	3000499:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ы
Т
%__inference_ou1_layer_call_fn_3000652

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCall╪
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_ou1_layer_call_and_return_conditional_losses_2998201o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000648:'#
!
_user_specified_name	3000646:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
ў
e
I__inference_activation_4_layer_call_and_return_conditional_losses_2998075

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :                  @]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Й
d
H__inference_rescaling_3_layer_call_and_return_conditional_losses_2998346

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
 *  А┐b
mulMulinputsCast/x:output:0*
T0*4
_output_shapes"
 :                  g
addAddV2mul:z:0Cast_1/x:output:0*
T0*4
_output_shapes"
 :                  \
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
╤
c
G__inference_activation_layer_call_and_return_conditional_losses_3000398

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:         @T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
у	
╥
%__inference_BPE_layer_call_fn_2998005
bpe_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallbpe_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_BPE_layer_call_and_return_conditional_losses_2997979|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2998001:'#
!
_user_specified_name	2997999:'#
!
_user_specified_name	2997997:'#
!
_user_specified_name	2997995:f b
4
_output_shapes"
 :                  
*
_user_specified_namebpe_layer1_input
╞
Є
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997807
spatial_layer1_input(
spatial_layer1_2997754:@$
spatial_layer1_2997756:@(
spatial_layer2_2997795:@@$
spatial_layer2_2997797:@
identityИв&spatial_layer1/StatefulPartitionedCallв&spatial_layer2/StatefulPartitionedCallа
&spatial_layer1/StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputspatial_layer1_2997754spatial_layer1_2997756*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_2997753э
activation/PartitionedCallPartitionedCall/spatial_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_2997763п
&spatial_layer2/StatefulPartitionedCallStatefulPartitionedCall#activation/PartitionedCall:output:0spatial_layer2_2997795spatial_layer2_2997797*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_2997794ё
activation_1/PartitionedCallPartitionedCall/spatial_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_2997804x
IdentityIdentity%activation_1/PartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @t
NoOpNoOp'^spatial_layer1/StatefulPartitionedCall'^spatial_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : 2P
&spatial_layer1/StatefulPartitionedCall&spatial_layer1/StatefulPartitionedCall2P
&spatial_layer2/StatefulPartitionedCall&spatial_layer2/StatefulPartitionedCall:'#
!
_user_specified_name	2997797:'#
!
_user_specified_name	2997795:'#
!
_user_specified_name	2997756:'#
!
_user_specified_name	2997754:a ]
+
_output_shapes
:         
.
_user_specified_namespatial_layer1_input
ё	
щ
8__inference_spatial_transformation_layer_call_fn_2997836
spatial_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997807s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2997832:'#
!
_user_specified_name	2997830:'#
!
_user_specified_name	2997828:'#
!
_user_specified_name	2997826:a ]
+
_output_shapes
:         
.
_user_specified_namespatial_layer1_input
╫
√
D__inference_dense_4_layer_call_and_return_conditional_losses_2998759

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
Г
Ц
)__inference_dense_1_layer_call_fn_2999908

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2998531s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2999904:'#
!
_user_specified_name	2999902:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
 
[
/__inference_concatenate_1_layer_call_fn_2999722
inputs_0
inputs_1
identity╥
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2998362m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*S
_input_shapesB
@:                  :                  :^Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_1:^ Z
4
_output_shapes"
 :                  
"
_user_specified_name
inputs_0
╣
r
H__inference_concatenate_layer_call_and_return_conditional_losses_2998354

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
:         W
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :OK
'
_output_shapes
:         
 
_user_specified_nameinputs:O K
'
_output_shapes
:         
 
_user_specified_nameinputs
мЛ
у
B__inference_model_layer_call_and_return_conditional_losses_2998797
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layer
bpe_2998378:@
bpe_2998380:@
bpe_2998382:@@
bpe_2998384:@
bve_2998387:@
bve_2998389:@
bve_2998391:@@
bve_2998393:@0
spatial_transformation_2998396:@,
spatial_transformation_2998398:@0
spatial_transformation_2998400:@@,
spatial_transformation_2998402:@2
multi_head_attention_2998441:@@.
multi_head_attention_2998443:@2
multi_head_attention_2998445:@@.
multi_head_attention_2998447:@2
multi_head_attention_2998449:@@.
multi_head_attention_2998451:@2
multi_head_attention_2998453:@@*
multi_head_attention_2998455:@
dense_2998496:@@
dense_2998498:@!
dense_1_2998532:@@
dense_1_2998534:@4
multi_head_attention_1_2998573:@@0
multi_head_attention_1_2998575:@4
multi_head_attention_1_2998577:@@0
multi_head_attention_1_2998579:@4
multi_head_attention_1_2998581:@@0
multi_head_attention_1_2998583:@4
multi_head_attention_1_2998585:@@,
multi_head_attention_1_2998587:@!
dense_2_2998628:@@
dense_2_2998630:@!
dense_3_2998664:@@
dense_3_2998666:@4
multi_head_attention_2_2998705:@@0
multi_head_attention_2_2998707:@4
multi_head_attention_2_2998709:@@0
multi_head_attention_2_2998711:@4
multi_head_attention_2_2998713:@@0
multi_head_attention_2_2998715:@4
multi_head_attention_2_2998717:@@,
multi_head_attention_2_2998719:@!
dense_4_2998760:@@
dense_4_2998762:@
	u_2998772:@@
	u_2998774:@
	u_2998776:@@
	u_2998778:@"
output_u_2998791:@
output_u_2998793:
identityИвBPE/StatefulPartitionedCallвBVE/StatefulPartitionedCallвU/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallв,multi_head_attention/StatefulPartitionedCallв.multi_head_attention_1/StatefulPartitionedCallв.multi_head_attention_2/StatefulPartitionedCallв output_u/StatefulPartitionedCallв.spatial_transformation/StatefulPartitionedCall┐
rescaling/PartitionedCallPartitionedCallx_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_2998319├
rescaling_1/PartitionedCallPartitionedCallt_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_2998328╥
rescaling_2/PartitionedCallPartitionedCall	xbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_2_layer_call_and_return_conditional_losses_2998337╥
rescaling_3/PartitionedCallPartitionedCall	tbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_3_layer_call_and_return_conditional_losses_2998346Е
concatenate/PartitionedCallPartitionedCall"rescaling/PartitionedCall:output:0$rescaling_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_2998354Ш
concatenate_1/PartitionedCallPartitionedCall$rescaling_2/PartitionedCall:output:0$rescaling_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2998362▄
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2998376н
BPE/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0bpe_2998378bpe_2998380bpe_2998382bpe_2998384*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_BPE_layer_call_and_return_conditional_losses_2997963Р
BVE/StatefulPartitionedCallStatefulPartitionedCall	ubc_layerbve_2998387bve_2998389bve_2998391bve_2998393*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_BVE_layer_call_and_return_conditional_losses_2998119Р
.spatial_transformation/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0spatial_transformation_2998396spatial_transformation_2998398spatial_transformation_2998400spatial_transformation_2998402*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997807щ
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:0$BVE/StatefulPartitionedCall:output:0$BPE/StatefulPartitionedCall:output:0multi_head_attention_2998441multi_head_attention_2998443multi_head_attention_2998445multi_head_attention_2998447multi_head_attention_2998449multi_head_attention_2998451multi_head_attention_2998453multi_head_attention_2998455*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2998440Я
add/PartitionedCallPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:05multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_2998463Д
dense/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0dense_2998496dense_2998498*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2998495Ц
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2998532dense_1_2998534*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2998531ю
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0$BVE/StatefulPartitionedCall:output:0$BPE/StatefulPartitionedCall:output:0multi_head_attention_1_2998573multi_head_attention_1_2998575multi_head_attention_1_2998577multi_head_attention_1_2998579multi_head_attention_1_2998581multi_head_attention_1_2998583multi_head_attention_1_2998585multi_head_attention_1_2998587*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_2998572Ц
add_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:07multi_head_attention_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_2998595О
dense_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0dense_2_2998628dense_2_2998630*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_2998627Ш
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_2998664dense_3_2998666*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_2998663ю
.multi_head_attention_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0$BVE/StatefulPartitionedCall:output:0$BPE/StatefulPartitionedCall:output:0multi_head_attention_2_2998705multi_head_attention_2_2998707multi_head_attention_2_2998709multi_head_attention_2_2998711multi_head_attention_2_2998713multi_head_attention_2_2998715multi_head_attention_2_2998717multi_head_attention_2_2998719*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_2998704Ц
add_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:07multi_head_attention_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_2998727О
dense_4/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0dense_4_2998760dense_4_2998762*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2998759▄
flatten/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2998770О
U/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	u_2998772	u_2998774	u_2998776	u_2998778*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_U_layer_call_and_return_conditional_losses_2998235Т
 output_u/StatefulPartitionedCallStatefulPartitionedCall"U/StatefulPartitionedCall:output:0output_u_2998791output_u_2998793*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_output_u_layer_call_and_return_conditional_losses_2998790x
IdentityIdentity)output_u/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         З
NoOpNoOp^BPE/StatefulPartitionedCall^BVE/StatefulPartitionedCall^U/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall/^multi_head_attention_2/StatefulPartitionedCall!^output_u/StatefulPartitionedCall/^spatial_transformation/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesё
ю:         :         :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
BPE/StatefulPartitionedCallBPE/StatefulPartitionedCall2:
BVE/StatefulPartitionedCallBVE/StatefulPartitionedCall26
U/StatefulPartitionedCallU/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2`
.multi_head_attention_2/StatefulPartitionedCall.multi_head_attention_2/StatefulPartitionedCall2D
 output_u/StatefulPartitionedCall output_u/StatefulPartitionedCall2`
.spatial_transformation/StatefulPartitionedCall.spatial_transformation/StatefulPartitionedCall:'8#
!
_user_specified_name	2998793:'7#
!
_user_specified_name	2998791:'6#
!
_user_specified_name	2998778:'5#
!
_user_specified_name	2998776:'4#
!
_user_specified_name	2998774:'3#
!
_user_specified_name	2998772:'2#
!
_user_specified_name	2998762:'1#
!
_user_specified_name	2998760:'0#
!
_user_specified_name	2998719:'/#
!
_user_specified_name	2998717:'.#
!
_user_specified_name	2998715:'-#
!
_user_specified_name	2998713:',#
!
_user_specified_name	2998711:'+#
!
_user_specified_name	2998709:'*#
!
_user_specified_name	2998707:')#
!
_user_specified_name	2998705:'(#
!
_user_specified_name	2998666:''#
!
_user_specified_name	2998664:'&#
!
_user_specified_name	2998630:'%#
!
_user_specified_name	2998628:'$#
!
_user_specified_name	2998587:'##
!
_user_specified_name	2998585:'"#
!
_user_specified_name	2998583:'!#
!
_user_specified_name	2998581:' #
!
_user_specified_name	2998579:'#
!
_user_specified_name	2998577:'#
!
_user_specified_name	2998575:'#
!
_user_specified_name	2998573:'#
!
_user_specified_name	2998534:'#
!
_user_specified_name	2998532:'#
!
_user_specified_name	2998498:'#
!
_user_specified_name	2998496:'#
!
_user_specified_name	2998455:'#
!
_user_specified_name	2998453:'#
!
_user_specified_name	2998451:'#
!
_user_specified_name	2998449:'#
!
_user_specified_name	2998447:'#
!
_user_specified_name	2998445:'#
!
_user_specified_name	2998443:'#
!
_user_specified_name	2998441:'#
!
_user_specified_name	2998402:'#
!
_user_specified_name	2998400:'#
!
_user_specified_name	2998398:'#
!
_user_specified_name	2998396:'#
!
_user_specified_name	2998393:'#
!
_user_specified_name	2998391:'
#
!
_user_specified_name	2998389:'	#
!
_user_specified_name	2998387:'#
!
_user_specified_name	2998384:'#
!
_user_specified_name	2998382:'#
!
_user_specified_name	2998380:'#
!
_user_specified_name	2998378:_[
4
_output_shapes"
 :                  
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:         
!
_user_specified_name	t_input:P L
'
_output_shapes
:         
!
_user_specified_name	x_input
╫
√
D__inference_dense_3_layer_call_and_return_conditional_losses_3000149

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
─
■
G__inference_bve_layer2_layer_call_and_return_conditional_losses_2998106

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 20
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
 :                  @
 
_user_specified_nameinputs
є	
ё
@__inference_ou2_layer_call_and_return_conditional_losses_2998222

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
и
E
)__inference_flatten_layer_call_fn_3000324

inputs
identity▓
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2998770`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
°	
Ў
E__inference_output_u_layer_call_and_return_conditional_losses_2998790

inputs0
matmul_readvariableop_resource:@-
biasadd_readvariableop_resource:
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         _
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
є	
ё
@__inference_ou1_layer_call_and_return_conditional_losses_2998201

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
╞
Ў
8__inference_multi_head_attention_2_layer_call_fn_3000172	
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
identityИвStatefulPartitionedCall╩
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_2998704s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	3000168:'	#
!
_user_specified_name	3000166:'#
!
_user_specified_name	3000164:'#
!
_user_specified_name	3000162:'#
!
_user_specified_name	3000160:'#
!
_user_specified_name	3000158:'#
!
_user_specified_name	3000156:'#
!
_user_specified_name	3000154:YU
4
_output_shapes"
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
╒
∙
B__inference_dense_layer_call_and_return_conditional_losses_2998495

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
ї
Ч
*__inference_output_u_layer_call_fn_3000339

inputs
unknown:@
	unknown_0:
identityИвStatefulPartitionedCall▌
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_output_u_layer_call_and_return_conditional_losses_2998790o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000335:'#
!
_user_specified_name	3000333:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╫
√
D__inference_dense_1_layer_call_and_return_conditional_losses_2998531

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
Х
├
@__inference_BVE_layer_call_and_return_conditional_losses_2998135
bve_layer1_input$
bve_layer1_2998122:@ 
bve_layer1_2998124:@$
bve_layer2_2998128:@@ 
bve_layer2_2998130:@
identityИв"bve_layer1/StatefulPartitionedCallв"bve_layer2/StatefulPartitionedCallХ
"bve_layer1/StatefulPartitionedCallStatefulPartitionedCallbve_layer1_inputbve_layer1_2998122bve_layer1_2998124*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bve_layer1_layer_call_and_return_conditional_losses_2998065Ў
activation_4/PartitionedCallPartitionedCall+bve_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_2998075к
"bve_layer2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0bve_layer2_2998128bve_layer2_2998130*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bve_layer2_layer_call_and_return_conditional_losses_2998106Ў
activation_5/PartitionedCallPartitionedCall+bve_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_2998116Б
IdentityIdentity%activation_5/PartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @l
NoOpNoOp#^bve_layer1/StatefulPartitionedCall#^bve_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 2H
"bve_layer1/StatefulPartitionedCall"bve_layer1/StatefulPartitionedCall2H
"bve_layer2/StatefulPartitionedCall"bve_layer2/StatefulPartitionedCall:'#
!
_user_specified_name	2998130:'#
!
_user_specified_name	2998128:'#
!
_user_specified_name	2998124:'#
!
_user_specified_name	2998122:f b
4
_output_shapes"
 :                  
*
_user_specified_namebve_layer1_input
Х
├
@__inference_BVE_layer_call_and_return_conditional_losses_2998119
bve_layer1_input$
bve_layer1_2998066:@ 
bve_layer1_2998068:@$
bve_layer2_2998107:@@ 
bve_layer2_2998109:@
identityИв"bve_layer1/StatefulPartitionedCallв"bve_layer2/StatefulPartitionedCallХ
"bve_layer1/StatefulPartitionedCallStatefulPartitionedCallbve_layer1_inputbve_layer1_2998066bve_layer1_2998068*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bve_layer1_layer_call_and_return_conditional_losses_2998065Ў
activation_4/PartitionedCallPartitionedCall+bve_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_4_layer_call_and_return_conditional_losses_2998075к
"bve_layer2/StatefulPartitionedCallStatefulPartitionedCall%activation_4/PartitionedCall:output:0bve_layer2_2998107bve_layer2_2998109*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bve_layer2_layer_call_and_return_conditional_losses_2998106Ў
activation_5/PartitionedCallPartitionedCall+bve_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_5_layer_call_and_return_conditional_losses_2998116Б
IdentityIdentity%activation_5/PartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @l
NoOpNoOp#^bve_layer1/StatefulPartitionedCall#^bve_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 2H
"bve_layer1/StatefulPartitionedCall"bve_layer1/StatefulPartitionedCall2H
"bve_layer2/StatefulPartitionedCall"bve_layer2/StatefulPartitionedCall:'#
!
_user_specified_name	2998109:'#
!
_user_specified_name	2998107:'#
!
_user_specified_name	2998068:'#
!
_user_specified_name	2998066:f b
4
_output_shapes"
 :                  
*
_user_specified_namebve_layer1_input
Й
d
H__inference_rescaling_2_layer_call_and_return_conditional_losses_2998337

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
 *  А┐b
mulMulinputsCast/x:output:0*
T0*4
_output_shapes"
 :                  g
addAddV2mul:z:0Cast_1/x:output:0*
T0*4
_output_shapes"
 :                  \
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
║
J
.__inference_activation_1_layer_call_fn_3000442

inputs
identity╗
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_1_layer_call_and_return_conditional_losses_2997804d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
у	
╥
%__inference_BPE_layer_call_fn_2997992
bpe_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityИвStatefulPartitionedCallЙ
StatefulPartitionedCallStatefulPartitionedCallbpe_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_BPE_layer_call_and_return_conditional_losses_2997963|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2997988:'#
!
_user_specified_name	2997986:'#
!
_user_specified_name	2997984:'#
!
_user_specified_name	2997982:f b
4
_output_shapes"
 :                  
*
_user_specified_namebpe_layer1_input
ЪС
╖2
 __inference__traced_save_3001039
file_prefix5
#read_disablecopyonread_dense_kernel:@@1
#read_1_disablecopyonread_dense_bias:@9
'read_2_disablecopyonread_dense_1_kernel:@@3
%read_3_disablecopyonread_dense_1_bias:@9
'read_4_disablecopyonread_dense_2_kernel:@@3
%read_5_disablecopyonread_dense_2_bias:@9
'read_6_disablecopyonread_dense_3_kernel:@@3
%read_7_disablecopyonread_dense_3_bias:@9
'read_8_disablecopyonread_dense_4_kernel:@@3
%read_9_disablecopyonread_dense_4_bias:@;
)read_10_disablecopyonread_output_u_kernel:@5
'read_11_disablecopyonread_output_u_bias:A
/read_12_disablecopyonread_spatial_layer1_kernel:@;
-read_13_disablecopyonread_spatial_layer1_bias:@A
/read_14_disablecopyonread_spatial_layer2_kernel:@@;
-read_15_disablecopyonread_spatial_layer2_bias:@=
+read_16_disablecopyonread_bpe_layer1_kernel:@7
)read_17_disablecopyonread_bpe_layer1_bias:@=
+read_18_disablecopyonread_bpe_layer2_kernel:@@7
)read_19_disablecopyonread_bpe_layer2_bias:@=
+read_20_disablecopyonread_bve_layer1_kernel:@7
)read_21_disablecopyonread_bve_layer1_bias:@=
+read_22_disablecopyonread_bve_layer2_kernel:@@7
)read_23_disablecopyonread_bve_layer2_bias:@Q
;read_24_disablecopyonread_multi_head_attention_query_kernel:@@K
9read_25_disablecopyonread_multi_head_attention_query_bias:@O
9read_26_disablecopyonread_multi_head_attention_key_kernel:@@I
7read_27_disablecopyonread_multi_head_attention_key_bias:@Q
;read_28_disablecopyonread_multi_head_attention_value_kernel:@@K
9read_29_disablecopyonread_multi_head_attention_value_bias:@\
Fread_30_disablecopyonread_multi_head_attention_attention_output_kernel:@@R
Dread_31_disablecopyonread_multi_head_attention_attention_output_bias:@S
=read_32_disablecopyonread_multi_head_attention_1_query_kernel:@@M
;read_33_disablecopyonread_multi_head_attention_1_query_bias:@Q
;read_34_disablecopyonread_multi_head_attention_1_key_kernel:@@K
9read_35_disablecopyonread_multi_head_attention_1_key_bias:@S
=read_36_disablecopyonread_multi_head_attention_1_value_kernel:@@M
;read_37_disablecopyonread_multi_head_attention_1_value_bias:@^
Hread_38_disablecopyonread_multi_head_attention_1_attention_output_kernel:@@T
Fread_39_disablecopyonread_multi_head_attention_1_attention_output_bias:@S
=read_40_disablecopyonread_multi_head_attention_2_query_kernel:@@M
;read_41_disablecopyonread_multi_head_attention_2_query_bias:@Q
;read_42_disablecopyonread_multi_head_attention_2_key_kernel:@@K
9read_43_disablecopyonread_multi_head_attention_2_key_bias:@S
=read_44_disablecopyonread_multi_head_attention_2_value_kernel:@@M
;read_45_disablecopyonread_multi_head_attention_2_value_bias:@^
Hread_46_disablecopyonread_multi_head_attention_2_attention_output_kernel:@@T
Fread_47_disablecopyonread_multi_head_attention_2_attention_output_bias:@6
$read_48_disablecopyonread_ou1_kernel:@@0
"read_49_disablecopyonread_ou1_bias:@6
$read_50_disablecopyonread_ou2_kernel:@@0
"read_51_disablecopyonread_ou2_bias:@
savev2_const
identity_105ИвMergeV2CheckpointsвRead/DisableCopyOnReadвRead/ReadVariableOpвRead_1/DisableCopyOnReadвRead_1/ReadVariableOpвRead_10/DisableCopyOnReadвRead_10/ReadVariableOpвRead_11/DisableCopyOnReadвRead_11/ReadVariableOpвRead_12/DisableCopyOnReadвRead_12/ReadVariableOpвRead_13/DisableCopyOnReadвRead_13/ReadVariableOpвRead_14/DisableCopyOnReadвRead_14/ReadVariableOpвRead_15/DisableCopyOnReadвRead_15/ReadVariableOpвRead_16/DisableCopyOnReadвRead_16/ReadVariableOpвRead_17/DisableCopyOnReadвRead_17/ReadVariableOpвRead_18/DisableCopyOnReadвRead_18/ReadVariableOpвRead_19/DisableCopyOnReadвRead_19/ReadVariableOpвRead_2/DisableCopyOnReadвRead_2/ReadVariableOpвRead_20/DisableCopyOnReadвRead_20/ReadVariableOpвRead_21/DisableCopyOnReadвRead_21/ReadVariableOpвRead_22/DisableCopyOnReadвRead_22/ReadVariableOpвRead_23/DisableCopyOnReadвRead_23/ReadVariableOpвRead_24/DisableCopyOnReadвRead_24/ReadVariableOpвRead_25/DisableCopyOnReadвRead_25/ReadVariableOpвRead_26/DisableCopyOnReadвRead_26/ReadVariableOpвRead_27/DisableCopyOnReadвRead_27/ReadVariableOpвRead_28/DisableCopyOnReadвRead_28/ReadVariableOpвRead_29/DisableCopyOnReadвRead_29/ReadVariableOpвRead_3/DisableCopyOnReadвRead_3/ReadVariableOpвRead_30/DisableCopyOnReadвRead_30/ReadVariableOpвRead_31/DisableCopyOnReadвRead_31/ReadVariableOpвRead_32/DisableCopyOnReadвRead_32/ReadVariableOpвRead_33/DisableCopyOnReadвRead_33/ReadVariableOpвRead_34/DisableCopyOnReadвRead_34/ReadVariableOpвRead_35/DisableCopyOnReadвRead_35/ReadVariableOpвRead_36/DisableCopyOnReadвRead_36/ReadVariableOpвRead_37/DisableCopyOnReadвRead_37/ReadVariableOpвRead_38/DisableCopyOnReadвRead_38/ReadVariableOpвRead_39/DisableCopyOnReadвRead_39/ReadVariableOpвRead_4/DisableCopyOnReadвRead_4/ReadVariableOpвRead_40/DisableCopyOnReadвRead_40/ReadVariableOpвRead_41/DisableCopyOnReadвRead_41/ReadVariableOpвRead_42/DisableCopyOnReadвRead_42/ReadVariableOpвRead_43/DisableCopyOnReadвRead_43/ReadVariableOpвRead_44/DisableCopyOnReadвRead_44/ReadVariableOpвRead_45/DisableCopyOnReadвRead_45/ReadVariableOpвRead_46/DisableCopyOnReadвRead_46/ReadVariableOpвRead_47/DisableCopyOnReadвRead_47/ReadVariableOpвRead_48/DisableCopyOnReadвRead_48/ReadVariableOpвRead_49/DisableCopyOnReadвRead_49/ReadVariableOpвRead_5/DisableCopyOnReadвRead_5/ReadVariableOpвRead_50/DisableCopyOnReadвRead_50/ReadVariableOpвRead_51/DisableCopyOnReadвRead_51/ReadVariableOpвRead_6/DisableCopyOnReadвRead_6/ReadVariableOpвRead_7/DisableCopyOnReadвRead_7/ReadVariableOpвRead_8/DisableCopyOnReadвRead_8/ReadVariableOpвRead_9/DisableCopyOnReadвRead_9/ReadVariableOpw
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
_temp/partБ
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
value	B : У
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: u
Read/DisableCopyOnReadDisableCopyOnRead#read_disablecopyonread_dense_kernel"/device:CPU:0*
_output_shapes
 Я
Read/ReadVariableOpReadVariableOp#read_disablecopyonread_dense_kernel^Read/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0i
IdentityIdentityRead/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@a

Identity_1IdentityIdentity:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_1/DisableCopyOnReadDisableCopyOnRead#read_1_disablecopyonread_dense_bias"/device:CPU:0*
_output_shapes
 Я
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
Read_2/DisableCopyOnReadDisableCopyOnRead'read_2_disablecopyonread_dense_1_kernel"/device:CPU:0*
_output_shapes
 з
Read_2/ReadVariableOpReadVariableOp'read_2_disablecopyonread_dense_1_kernel^Read_2/DisableCopyOnRead"/device:CPU:0*
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

:@@y
Read_3/DisableCopyOnReadDisableCopyOnRead%read_3_disablecopyonread_dense_1_bias"/device:CPU:0*
_output_shapes
 б
Read_3/ReadVariableOpReadVariableOp%read_3_disablecopyonread_dense_1_bias^Read_3/DisableCopyOnRead"/device:CPU:0*
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
Read_4/DisableCopyOnReadDisableCopyOnRead'read_4_disablecopyonread_dense_2_kernel"/device:CPU:0*
_output_shapes
 з
Read_4/ReadVariableOpReadVariableOp'read_4_disablecopyonread_dense_2_kernel^Read_4/DisableCopyOnRead"/device:CPU:0*
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
Read_5/DisableCopyOnReadDisableCopyOnRead%read_5_disablecopyonread_dense_2_bias"/device:CPU:0*
_output_shapes
 б
Read_5/ReadVariableOpReadVariableOp%read_5_disablecopyonread_dense_2_bias^Read_5/DisableCopyOnRead"/device:CPU:0*
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
 з
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
 б
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
 з
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
 б
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
 л
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
 е
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
:Д
Read_12/DisableCopyOnReadDisableCopyOnRead/read_12_disablecopyonread_spatial_layer1_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_12/ReadVariableOpReadVariableOp/read_12_disablecopyonread_spatial_layer1_kernel^Read_12/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_24IdentityRead_12/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_25IdentityIdentity_24:output:0"/device:CPU:0*
T0*
_output_shapes

:@В
Read_13/DisableCopyOnReadDisableCopyOnRead-read_13_disablecopyonread_spatial_layer1_bias"/device:CPU:0*
_output_shapes
 л
Read_13/ReadVariableOpReadVariableOp-read_13_disablecopyonread_spatial_layer1_bias^Read_13/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_26IdentityRead_13/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_27IdentityIdentity_26:output:0"/device:CPU:0*
T0*
_output_shapes
:@Д
Read_14/DisableCopyOnReadDisableCopyOnRead/read_14_disablecopyonread_spatial_layer2_kernel"/device:CPU:0*
_output_shapes
 ▒
Read_14/ReadVariableOpReadVariableOp/read_14_disablecopyonread_spatial_layer2_kernel^Read_14/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_28IdentityRead_14/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_29IdentityIdentity_28:output:0"/device:CPU:0*
T0*
_output_shapes

:@@В
Read_15/DisableCopyOnReadDisableCopyOnRead-read_15_disablecopyonread_spatial_layer2_bias"/device:CPU:0*
_output_shapes
 л
Read_15/ReadVariableOpReadVariableOp-read_15_disablecopyonread_spatial_layer2_bias^Read_15/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_30IdentityRead_15/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_31IdentityIdentity_30:output:0"/device:CPU:0*
T0*
_output_shapes
:@А
Read_16/DisableCopyOnReadDisableCopyOnRead+read_16_disablecopyonread_bpe_layer1_kernel"/device:CPU:0*
_output_shapes
 н
Read_16/ReadVariableOpReadVariableOp+read_16_disablecopyonread_bpe_layer1_kernel^Read_16/DisableCopyOnRead"/device:CPU:0*
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

:@~
Read_17/DisableCopyOnReadDisableCopyOnRead)read_17_disablecopyonread_bpe_layer1_bias"/device:CPU:0*
_output_shapes
 з
Read_17/ReadVariableOpReadVariableOp)read_17_disablecopyonread_bpe_layer1_bias^Read_17/DisableCopyOnRead"/device:CPU:0*
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
:@А
Read_18/DisableCopyOnReadDisableCopyOnRead+read_18_disablecopyonread_bpe_layer2_kernel"/device:CPU:0*
_output_shapes
 н
Read_18/ReadVariableOpReadVariableOp+read_18_disablecopyonread_bpe_layer2_kernel^Read_18/DisableCopyOnRead"/device:CPU:0*
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

:@@~
Read_19/DisableCopyOnReadDisableCopyOnRead)read_19_disablecopyonread_bpe_layer2_bias"/device:CPU:0*
_output_shapes
 з
Read_19/ReadVariableOpReadVariableOp)read_19_disablecopyonread_bpe_layer2_bias^Read_19/DisableCopyOnRead"/device:CPU:0*
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
:@А
Read_20/DisableCopyOnReadDisableCopyOnRead+read_20_disablecopyonread_bve_layer1_kernel"/device:CPU:0*
_output_shapes
 н
Read_20/ReadVariableOpReadVariableOp+read_20_disablecopyonread_bve_layer1_kernel^Read_20/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_40IdentityRead_20/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_41IdentityIdentity_40:output:0"/device:CPU:0*
T0*
_output_shapes

:@~
Read_21/DisableCopyOnReadDisableCopyOnRead)read_21_disablecopyonread_bve_layer1_bias"/device:CPU:0*
_output_shapes
 з
Read_21/ReadVariableOpReadVariableOp)read_21_disablecopyonread_bve_layer1_bias^Read_21/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_42IdentityRead_21/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_43IdentityIdentity_42:output:0"/device:CPU:0*
T0*
_output_shapes
:@А
Read_22/DisableCopyOnReadDisableCopyOnRead+read_22_disablecopyonread_bve_layer2_kernel"/device:CPU:0*
_output_shapes
 н
Read_22/ReadVariableOpReadVariableOp+read_22_disablecopyonread_bve_layer2_kernel^Read_22/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_44IdentityRead_22/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_45IdentityIdentity_44:output:0"/device:CPU:0*
T0*
_output_shapes

:@@~
Read_23/DisableCopyOnReadDisableCopyOnRead)read_23_disablecopyonread_bve_layer2_bias"/device:CPU:0*
_output_shapes
 з
Read_23/ReadVariableOpReadVariableOp)read_23_disablecopyonread_bve_layer2_bias^Read_23/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_46IdentityRead_23/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_47IdentityIdentity_46:output:0"/device:CPU:0*
T0*
_output_shapes
:@Р
Read_24/DisableCopyOnReadDisableCopyOnRead;read_24_disablecopyonread_multi_head_attention_query_kernel"/device:CPU:0*
_output_shapes
 ┴
Read_24/ReadVariableOpReadVariableOp;read_24_disablecopyonread_multi_head_attention_query_kernel^Read_24/DisableCopyOnRead"/device:CPU:0*"
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
:@@О
Read_25/DisableCopyOnReadDisableCopyOnRead9read_25_disablecopyonread_multi_head_attention_query_bias"/device:CPU:0*
_output_shapes
 ╗
Read_25/ReadVariableOpReadVariableOp9read_25_disablecopyonread_multi_head_attention_query_bias^Read_25/DisableCopyOnRead"/device:CPU:0*
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

:@О
Read_26/DisableCopyOnReadDisableCopyOnRead9read_26_disablecopyonread_multi_head_attention_key_kernel"/device:CPU:0*
_output_shapes
 ┐
Read_26/ReadVariableOpReadVariableOp9read_26_disablecopyonread_multi_head_attention_key_kernel^Read_26/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_52IdentityRead_26/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_53IdentityIdentity_52:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@М
Read_27/DisableCopyOnReadDisableCopyOnRead7read_27_disablecopyonread_multi_head_attention_key_bias"/device:CPU:0*
_output_shapes
 ╣
Read_27/ReadVariableOpReadVariableOp7read_27_disablecopyonread_multi_head_attention_key_bias^Read_27/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_54IdentityRead_27/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_55IdentityIdentity_54:output:0"/device:CPU:0*
T0*
_output_shapes

:@Р
Read_28/DisableCopyOnReadDisableCopyOnRead;read_28_disablecopyonread_multi_head_attention_value_kernel"/device:CPU:0*
_output_shapes
 ┴
Read_28/ReadVariableOpReadVariableOp;read_28_disablecopyonread_multi_head_attention_value_kernel^Read_28/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_56IdentityRead_28/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_57IdentityIdentity_56:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@О
Read_29/DisableCopyOnReadDisableCopyOnRead9read_29_disablecopyonread_multi_head_attention_value_bias"/device:CPU:0*
_output_shapes
 ╗
Read_29/ReadVariableOpReadVariableOp9read_29_disablecopyonread_multi_head_attention_value_bias^Read_29/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_58IdentityRead_29/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_59IdentityIdentity_58:output:0"/device:CPU:0*
T0*
_output_shapes

:@Ы
Read_30/DisableCopyOnReadDisableCopyOnReadFread_30_disablecopyonread_multi_head_attention_attention_output_kernel"/device:CPU:0*
_output_shapes
 ╠
Read_30/ReadVariableOpReadVariableOpFread_30_disablecopyonread_multi_head_attention_attention_output_kernel^Read_30/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_60IdentityRead_30/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_61IdentityIdentity_60:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@Щ
Read_31/DisableCopyOnReadDisableCopyOnReadDread_31_disablecopyonread_multi_head_attention_attention_output_bias"/device:CPU:0*
_output_shapes
 ┬
Read_31/ReadVariableOpReadVariableOpDread_31_disablecopyonread_multi_head_attention_attention_output_bias^Read_31/DisableCopyOnRead"/device:CPU:0*
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
:@Т
Read_32/DisableCopyOnReadDisableCopyOnRead=read_32_disablecopyonread_multi_head_attention_1_query_kernel"/device:CPU:0*
_output_shapes
 ├
Read_32/ReadVariableOpReadVariableOp=read_32_disablecopyonread_multi_head_attention_1_query_kernel^Read_32/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_64IdentityRead_32/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_65IdentityIdentity_64:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@Р
Read_33/DisableCopyOnReadDisableCopyOnRead;read_33_disablecopyonread_multi_head_attention_1_query_bias"/device:CPU:0*
_output_shapes
 ╜
Read_33/ReadVariableOpReadVariableOp;read_33_disablecopyonread_multi_head_attention_1_query_bias^Read_33/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_66IdentityRead_33/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_67IdentityIdentity_66:output:0"/device:CPU:0*
T0*
_output_shapes

:@Р
Read_34/DisableCopyOnReadDisableCopyOnRead;read_34_disablecopyonread_multi_head_attention_1_key_kernel"/device:CPU:0*
_output_shapes
 ┴
Read_34/ReadVariableOpReadVariableOp;read_34_disablecopyonread_multi_head_attention_1_key_kernel^Read_34/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_68IdentityRead_34/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_69IdentityIdentity_68:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@О
Read_35/DisableCopyOnReadDisableCopyOnRead9read_35_disablecopyonread_multi_head_attention_1_key_bias"/device:CPU:0*
_output_shapes
 ╗
Read_35/ReadVariableOpReadVariableOp9read_35_disablecopyonread_multi_head_attention_1_key_bias^Read_35/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_70IdentityRead_35/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_71IdentityIdentity_70:output:0"/device:CPU:0*
T0*
_output_shapes

:@Т
Read_36/DisableCopyOnReadDisableCopyOnRead=read_36_disablecopyonread_multi_head_attention_1_value_kernel"/device:CPU:0*
_output_shapes
 ├
Read_36/ReadVariableOpReadVariableOp=read_36_disablecopyonread_multi_head_attention_1_value_kernel^Read_36/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_72IdentityRead_36/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_73IdentityIdentity_72:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@Р
Read_37/DisableCopyOnReadDisableCopyOnRead;read_37_disablecopyonread_multi_head_attention_1_value_bias"/device:CPU:0*
_output_shapes
 ╜
Read_37/ReadVariableOpReadVariableOp;read_37_disablecopyonread_multi_head_attention_1_value_bias^Read_37/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_74IdentityRead_37/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_75IdentityIdentity_74:output:0"/device:CPU:0*
T0*
_output_shapes

:@Э
Read_38/DisableCopyOnReadDisableCopyOnReadHread_38_disablecopyonread_multi_head_attention_1_attention_output_kernel"/device:CPU:0*
_output_shapes
 ╬
Read_38/ReadVariableOpReadVariableOpHread_38_disablecopyonread_multi_head_attention_1_attention_output_kernel^Read_38/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_76IdentityRead_38/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_77IdentityIdentity_76:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@Ы
Read_39/DisableCopyOnReadDisableCopyOnReadFread_39_disablecopyonread_multi_head_attention_1_attention_output_bias"/device:CPU:0*
_output_shapes
 ─
Read_39/ReadVariableOpReadVariableOpFread_39_disablecopyonread_multi_head_attention_1_attention_output_bias^Read_39/DisableCopyOnRead"/device:CPU:0*
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
:@Т
Read_40/DisableCopyOnReadDisableCopyOnRead=read_40_disablecopyonread_multi_head_attention_2_query_kernel"/device:CPU:0*
_output_shapes
 ├
Read_40/ReadVariableOpReadVariableOp=read_40_disablecopyonread_multi_head_attention_2_query_kernel^Read_40/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_80IdentityRead_40/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_81IdentityIdentity_80:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@Р
Read_41/DisableCopyOnReadDisableCopyOnRead;read_41_disablecopyonread_multi_head_attention_2_query_bias"/device:CPU:0*
_output_shapes
 ╜
Read_41/ReadVariableOpReadVariableOp;read_41_disablecopyonread_multi_head_attention_2_query_bias^Read_41/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_82IdentityRead_41/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_83IdentityIdentity_82:output:0"/device:CPU:0*
T0*
_output_shapes

:@Р
Read_42/DisableCopyOnReadDisableCopyOnRead;read_42_disablecopyonread_multi_head_attention_2_key_kernel"/device:CPU:0*
_output_shapes
 ┴
Read_42/ReadVariableOpReadVariableOp;read_42_disablecopyonread_multi_head_attention_2_key_kernel^Read_42/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_84IdentityRead_42/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_85IdentityIdentity_84:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@О
Read_43/DisableCopyOnReadDisableCopyOnRead9read_43_disablecopyonread_multi_head_attention_2_key_bias"/device:CPU:0*
_output_shapes
 ╗
Read_43/ReadVariableOpReadVariableOp9read_43_disablecopyonread_multi_head_attention_2_key_bias^Read_43/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_86IdentityRead_43/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_87IdentityIdentity_86:output:0"/device:CPU:0*
T0*
_output_shapes

:@Т
Read_44/DisableCopyOnReadDisableCopyOnRead=read_44_disablecopyonread_multi_head_attention_2_value_kernel"/device:CPU:0*
_output_shapes
 ├
Read_44/ReadVariableOpReadVariableOp=read_44_disablecopyonread_multi_head_attention_2_value_kernel^Read_44/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_88IdentityRead_44/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_89IdentityIdentity_88:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@Р
Read_45/DisableCopyOnReadDisableCopyOnRead;read_45_disablecopyonread_multi_head_attention_2_value_bias"/device:CPU:0*
_output_shapes
 ╜
Read_45/ReadVariableOpReadVariableOp;read_45_disablecopyonread_multi_head_attention_2_value_bias^Read_45/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@*
dtype0o
Identity_90IdentityRead_45/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@e
Identity_91IdentityIdentity_90:output:0"/device:CPU:0*
T0*
_output_shapes

:@Э
Read_46/DisableCopyOnReadDisableCopyOnReadHread_46_disablecopyonread_multi_head_attention_2_attention_output_kernel"/device:CPU:0*
_output_shapes
 ╬
Read_46/ReadVariableOpReadVariableOpHread_46_disablecopyonread_multi_head_attention_2_attention_output_kernel^Read_46/DisableCopyOnRead"/device:CPU:0*"
_output_shapes
:@@*
dtype0s
Identity_92IdentityRead_46/ReadVariableOp:value:0"/device:CPU:0*
T0*"
_output_shapes
:@@i
Identity_93IdentityIdentity_92:output:0"/device:CPU:0*
T0*"
_output_shapes
:@@Ы
Read_47/DisableCopyOnReadDisableCopyOnReadFread_47_disablecopyonread_multi_head_attention_2_attention_output_bias"/device:CPU:0*
_output_shapes
 ─
Read_47/ReadVariableOpReadVariableOpFread_47_disablecopyonread_multi_head_attention_2_attention_output_bias^Read_47/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_94IdentityRead_47/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_95IdentityIdentity_94:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_48/DisableCopyOnReadDisableCopyOnRead$read_48_disablecopyonread_ou1_kernel"/device:CPU:0*
_output_shapes
 ж
Read_48/ReadVariableOpReadVariableOp$read_48_disablecopyonread_ou1_kernel^Read_48/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0o
Identity_96IdentityRead_48/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@e
Identity_97IdentityIdentity_96:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_49/DisableCopyOnReadDisableCopyOnRead"read_49_disablecopyonread_ou1_bias"/device:CPU:0*
_output_shapes
 а
Read_49/ReadVariableOpReadVariableOp"read_49_disablecopyonread_ou1_bias^Read_49/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0k
Identity_98IdentityRead_49/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@a
Identity_99IdentityIdentity_98:output:0"/device:CPU:0*
T0*
_output_shapes
:@y
Read_50/DisableCopyOnReadDisableCopyOnRead$read_50_disablecopyonread_ou2_kernel"/device:CPU:0*
_output_shapes
 ж
Read_50/ReadVariableOpReadVariableOp$read_50_disablecopyonread_ou2_kernel^Read_50/DisableCopyOnRead"/device:CPU:0*
_output_shapes

:@@*
dtype0p
Identity_100IdentityRead_50/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes

:@@g
Identity_101IdentityIdentity_100:output:0"/device:CPU:0*
T0*
_output_shapes

:@@w
Read_51/DisableCopyOnReadDisableCopyOnRead"read_51_disablecopyonread_ou2_bias"/device:CPU:0*
_output_shapes
 а
Read_51/ReadVariableOpReadVariableOp"read_51_disablecopyonread_ou2_bias^Read_51/DisableCopyOnRead"/device:CPU:0*
_output_shapes
:@*
dtype0l
Identity_102IdentityRead_51/ReadVariableOp:value:0"/device:CPU:0*
T0*
_output_shapes
:@c
Identity_103IdentityIdentity_102:output:0"/device:CPU:0*
T0*
_output_shapes
:@А
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*й
valueЯBЬ5B6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-5/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-5/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-7/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-7/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB&variables/0/.ATTRIBUTES/VARIABLE_VALUEB&variables/1/.ATTRIBUTES/VARIABLE_VALUEB&variables/2/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB&variables/5/.ATTRIBUTES/VARIABLE_VALUEB&variables/6/.ATTRIBUTES/VARIABLE_VALUEB&variables/7/.ATTRIBUTES/VARIABLE_VALUEB&variables/8/.ATTRIBUTES/VARIABLE_VALUEB&variables/9/.ATTRIBUTES/VARIABLE_VALUEB'variables/10/.ATTRIBUTES/VARIABLE_VALUEB'variables/11/.ATTRIBUTES/VARIABLE_VALUEB'variables/12/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB'variables/15/.ATTRIBUTES/VARIABLE_VALUEB'variables/16/.ATTRIBUTES/VARIABLE_VALUEB'variables/17/.ATTRIBUTES/VARIABLE_VALUEB'variables/18/.ATTRIBUTES/VARIABLE_VALUEB'variables/19/.ATTRIBUTES/VARIABLE_VALUEB'variables/24/.ATTRIBUTES/VARIABLE_VALUEB'variables/25/.ATTRIBUTES/VARIABLE_VALUEB'variables/26/.ATTRIBUTES/VARIABLE_VALUEB'variables/27/.ATTRIBUTES/VARIABLE_VALUEB'variables/28/.ATTRIBUTES/VARIABLE_VALUEB'variables/29/.ATTRIBUTES/VARIABLE_VALUEB'variables/30/.ATTRIBUTES/VARIABLE_VALUEB'variables/31/.ATTRIBUTES/VARIABLE_VALUEB'variables/36/.ATTRIBUTES/VARIABLE_VALUEB'variables/37/.ATTRIBUTES/VARIABLE_VALUEB'variables/38/.ATTRIBUTES/VARIABLE_VALUEB'variables/39/.ATTRIBUTES/VARIABLE_VALUEB'variables/40/.ATTRIBUTES/VARIABLE_VALUEB'variables/41/.ATTRIBUTES/VARIABLE_VALUEB'variables/42/.ATTRIBUTES/VARIABLE_VALUEB'variables/43/.ATTRIBUTES/VARIABLE_VALUEB'variables/46/.ATTRIBUTES/VARIABLE_VALUEB'variables/47/.ATTRIBUTES/VARIABLE_VALUEB'variables/48/.ATTRIBUTES/VARIABLE_VALUEB'variables/49/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH╫
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:5*
dtype0*}
valuetBr5B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B Б
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0Identity_1:output:0Identity_3:output:0Identity_5:output:0Identity_7:output:0Identity_9:output:0Identity_11:output:0Identity_13:output:0Identity_15:output:0Identity_17:output:0Identity_19:output:0Identity_21:output:0Identity_23:output:0Identity_25:output:0Identity_27:output:0Identity_29:output:0Identity_31:output:0Identity_33:output:0Identity_35:output:0Identity_37:output:0Identity_39:output:0Identity_41:output:0Identity_43:output:0Identity_45:output:0Identity_47:output:0Identity_49:output:0Identity_51:output:0Identity_53:output:0Identity_55:output:0Identity_57:output:0Identity_59:output:0Identity_61:output:0Identity_63:output:0Identity_65:output:0Identity_67:output:0Identity_69:output:0Identity_71:output:0Identity_73:output:0Identity_75:output:0Identity_77:output:0Identity_79:output:0Identity_81:output:0Identity_83:output:0Identity_85:output:0Identity_87:output:0Identity_89:output:0Identity_91:output:0Identity_93:output:0Identity_95:output:0Identity_97:output:0Identity_99:output:0Identity_101:output:0Identity_103:output:0savev2_const"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 *C
dtypes9
725Р
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:│
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*&
 _has_manual_control_dependencies(*
_output_shapes
 j
Identity_104Identityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: W
Identity_105IdentityIdentity_104:output:0^NoOp*
T0*
_output_shapes
: у
NoOpNoOp^MergeV2Checkpoints^Read/DisableCopyOnRead^Read/ReadVariableOp^Read_1/DisableCopyOnRead^Read_1/ReadVariableOp^Read_10/DisableCopyOnRead^Read_10/ReadVariableOp^Read_11/DisableCopyOnRead^Read_11/ReadVariableOp^Read_12/DisableCopyOnRead^Read_12/ReadVariableOp^Read_13/DisableCopyOnRead^Read_13/ReadVariableOp^Read_14/DisableCopyOnRead^Read_14/ReadVariableOp^Read_15/DisableCopyOnRead^Read_15/ReadVariableOp^Read_16/DisableCopyOnRead^Read_16/ReadVariableOp^Read_17/DisableCopyOnRead^Read_17/ReadVariableOp^Read_18/DisableCopyOnRead^Read_18/ReadVariableOp^Read_19/DisableCopyOnRead^Read_19/ReadVariableOp^Read_2/DisableCopyOnRead^Read_2/ReadVariableOp^Read_20/DisableCopyOnRead^Read_20/ReadVariableOp^Read_21/DisableCopyOnRead^Read_21/ReadVariableOp^Read_22/DisableCopyOnRead^Read_22/ReadVariableOp^Read_23/DisableCopyOnRead^Read_23/ReadVariableOp^Read_24/DisableCopyOnRead^Read_24/ReadVariableOp^Read_25/DisableCopyOnRead^Read_25/ReadVariableOp^Read_26/DisableCopyOnRead^Read_26/ReadVariableOp^Read_27/DisableCopyOnRead^Read_27/ReadVariableOp^Read_28/DisableCopyOnRead^Read_28/ReadVariableOp^Read_29/DisableCopyOnRead^Read_29/ReadVariableOp^Read_3/DisableCopyOnRead^Read_3/ReadVariableOp^Read_30/DisableCopyOnRead^Read_30/ReadVariableOp^Read_31/DisableCopyOnRead^Read_31/ReadVariableOp^Read_32/DisableCopyOnRead^Read_32/ReadVariableOp^Read_33/DisableCopyOnRead^Read_33/ReadVariableOp^Read_34/DisableCopyOnRead^Read_34/ReadVariableOp^Read_35/DisableCopyOnRead^Read_35/ReadVariableOp^Read_36/DisableCopyOnRead^Read_36/ReadVariableOp^Read_37/DisableCopyOnRead^Read_37/ReadVariableOp^Read_38/DisableCopyOnRead^Read_38/ReadVariableOp^Read_39/DisableCopyOnRead^Read_39/ReadVariableOp^Read_4/DisableCopyOnRead^Read_4/ReadVariableOp^Read_40/DisableCopyOnRead^Read_40/ReadVariableOp^Read_41/DisableCopyOnRead^Read_41/ReadVariableOp^Read_42/DisableCopyOnRead^Read_42/ReadVariableOp^Read_43/DisableCopyOnRead^Read_43/ReadVariableOp^Read_44/DisableCopyOnRead^Read_44/ReadVariableOp^Read_45/DisableCopyOnRead^Read_45/ReadVariableOp^Read_46/DisableCopyOnRead^Read_46/ReadVariableOp^Read_47/DisableCopyOnRead^Read_47/ReadVariableOp^Read_48/DisableCopyOnRead^Read_48/ReadVariableOp^Read_49/DisableCopyOnRead^Read_49/ReadVariableOp^Read_5/DisableCopyOnRead^Read_5/ReadVariableOp^Read_50/DisableCopyOnRead^Read_50/ReadVariableOp^Read_51/DisableCopyOnRead^Read_51/ReadVariableOp^Read_6/DisableCopyOnRead^Read_6/ReadVariableOp^Read_7/DisableCopyOnRead^Read_7/ReadVariableOp^Read_8/DisableCopyOnRead^Read_8/ReadVariableOp^Read_9/DisableCopyOnRead^Read_9/ReadVariableOp*
_output_shapes
 "%
identity_105Identity_105:output:0*(
_construction_contextkEagerRuntime*
_input_shapesn
l: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2(
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
Read_4/ReadVariableOpRead_4/ReadVariableOp26
Read_40/DisableCopyOnReadRead_40/DisableCopyOnRead20
Read_40/ReadVariableOpRead_40/ReadVariableOp26
Read_41/DisableCopyOnReadRead_41/DisableCopyOnRead20
Read_41/ReadVariableOpRead_41/ReadVariableOp26
Read_42/DisableCopyOnReadRead_42/DisableCopyOnRead20
Read_42/ReadVariableOpRead_42/ReadVariableOp26
Read_43/DisableCopyOnReadRead_43/DisableCopyOnRead20
Read_43/ReadVariableOpRead_43/ReadVariableOp26
Read_44/DisableCopyOnReadRead_44/DisableCopyOnRead20
Read_44/ReadVariableOpRead_44/ReadVariableOp26
Read_45/DisableCopyOnReadRead_45/DisableCopyOnRead20
Read_45/ReadVariableOpRead_45/ReadVariableOp26
Read_46/DisableCopyOnReadRead_46/DisableCopyOnRead20
Read_46/ReadVariableOpRead_46/ReadVariableOp26
Read_47/DisableCopyOnReadRead_47/DisableCopyOnRead20
Read_47/ReadVariableOpRead_47/ReadVariableOp26
Read_48/DisableCopyOnReadRead_48/DisableCopyOnRead20
Read_48/ReadVariableOpRead_48/ReadVariableOp26
Read_49/DisableCopyOnReadRead_49/DisableCopyOnRead20
Read_49/ReadVariableOpRead_49/ReadVariableOp24
Read_5/DisableCopyOnReadRead_5/DisableCopyOnRead2.
Read_5/ReadVariableOpRead_5/ReadVariableOp26
Read_50/DisableCopyOnReadRead_50/DisableCopyOnRead20
Read_50/ReadVariableOpRead_50/ReadVariableOp26
Read_51/DisableCopyOnReadRead_51/DisableCopyOnRead20
Read_51/ReadVariableOpRead_51/ReadVariableOp24
Read_6/DisableCopyOnReadRead_6/DisableCopyOnRead2.
Read_6/ReadVariableOpRead_6/ReadVariableOp24
Read_7/DisableCopyOnReadRead_7/DisableCopyOnRead2.
Read_7/ReadVariableOpRead_7/ReadVariableOp24
Read_8/DisableCopyOnReadRead_8/DisableCopyOnRead2.
Read_8/ReadVariableOpRead_8/ReadVariableOp24
Read_9/DisableCopyOnReadRead_9/DisableCopyOnRead2.
Read_9/ReadVariableOpRead_9/ReadVariableOp:=59

_output_shapes
: 

_user_specified_nameConst:(4$
"
_user_specified_name
ou2/bias:*3&
$
_user_specified_name
ou2/kernel:(2$
"
_user_specified_name
ou1/bias:*1&
$
_user_specified_name
ou1/kernel:L0H
F
_user_specified_name.,multi_head_attention_2/attention_output/bias:N/J
H
_user_specified_name0.multi_head_attention_2/attention_output/kernel:A.=
;
_user_specified_name#!multi_head_attention_2/value/bias:C-?
=
_user_specified_name%#multi_head_attention_2/value/kernel:?,;
9
_user_specified_name!multi_head_attention_2/key/bias:A+=
;
_user_specified_name#!multi_head_attention_2/key/kernel:A*=
;
_user_specified_name#!multi_head_attention_2/query/bias:C)?
=
_user_specified_name%#multi_head_attention_2/query/kernel:L(H
F
_user_specified_name.,multi_head_attention_1/attention_output/bias:N'J
H
_user_specified_name0.multi_head_attention_1/attention_output/kernel:A&=
;
_user_specified_name#!multi_head_attention_1/value/bias:C%?
=
_user_specified_name%#multi_head_attention_1/value/kernel:?$;
9
_user_specified_name!multi_head_attention_1/key/bias:A#=
;
_user_specified_name#!multi_head_attention_1/key/kernel:A"=
;
_user_specified_name#!multi_head_attention_1/query/bias:C!?
=
_user_specified_name%#multi_head_attention_1/query/kernel:J F
D
_user_specified_name,*multi_head_attention/attention_output/bias:LH
F
_user_specified_name.,multi_head_attention/attention_output/kernel:?;
9
_user_specified_name!multi_head_attention/value/bias:A=
;
_user_specified_name#!multi_head_attention/value/kernel:=9
7
_user_specified_namemulti_head_attention/key/bias:?;
9
_user_specified_name!multi_head_attention/key/kernel:?;
9
_user_specified_name!multi_head_attention/query/bias:A=
;
_user_specified_name#!multi_head_attention/query/kernel:/+
)
_user_specified_namebve_layer2/bias:1-
+
_user_specified_namebve_layer2/kernel:/+
)
_user_specified_namebve_layer1/bias:1-
+
_user_specified_namebve_layer1/kernel:/+
)
_user_specified_namebpe_layer2/bias:1-
+
_user_specified_namebpe_layer2/kernel:/+
)
_user_specified_namebpe_layer1/bias:1-
+
_user_specified_namebpe_layer1/kernel:3/
-
_user_specified_namespatial_layer2/bias:51
/
_user_specified_namespatial_layer2/kernel:3/
-
_user_specified_namespatial_layer1/bias:51
/
_user_specified_namespatial_layer1/kernel:-)
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
_user_specified_namedense_2/bias:.*
(
_user_specified_namedense_2/kernel:,(
&
_user_specified_namedense_1/bias:.*
(
_user_specified_namedense_1/kernel:*&
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
б.
Т
'__inference_model_layer_call_fn_2999154
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layer
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
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

unknown_23:@@

unknown_24:@ 

unknown_25:@@

unknown_26:@ 

unknown_27:@@

unknown_28:@ 

unknown_29:@@

unknown_30:@

unknown_31:@@

unknown_32:@

unknown_33:@@

unknown_34:@ 

unknown_35:@@

unknown_36:@ 

unknown_37:@@

unknown_38:@ 

unknown_39:@@

unknown_40:@ 

unknown_41:@@

unknown_42:@

unknown_43:@@

unknown_44:@

unknown_45:@@

unknown_46:@

unknown_47:@@

unknown_48:@

unknown_49:@

unknown_50:
identityИвStatefulPartitionedCall╝
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
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_model_layer_call_and_return_conditional_losses_2998797o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesё
ю:         :         :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'8#
!
_user_specified_name	2999150:'7#
!
_user_specified_name	2999148:'6#
!
_user_specified_name	2999146:'5#
!
_user_specified_name	2999144:'4#
!
_user_specified_name	2999142:'3#
!
_user_specified_name	2999140:'2#
!
_user_specified_name	2999138:'1#
!
_user_specified_name	2999136:'0#
!
_user_specified_name	2999134:'/#
!
_user_specified_name	2999132:'.#
!
_user_specified_name	2999130:'-#
!
_user_specified_name	2999128:',#
!
_user_specified_name	2999126:'+#
!
_user_specified_name	2999124:'*#
!
_user_specified_name	2999122:')#
!
_user_specified_name	2999120:'(#
!
_user_specified_name	2999118:''#
!
_user_specified_name	2999116:'&#
!
_user_specified_name	2999114:'%#
!
_user_specified_name	2999112:'$#
!
_user_specified_name	2999110:'##
!
_user_specified_name	2999108:'"#
!
_user_specified_name	2999106:'!#
!
_user_specified_name	2999104:' #
!
_user_specified_name	2999102:'#
!
_user_specified_name	2999100:'#
!
_user_specified_name	2999098:'#
!
_user_specified_name	2999096:'#
!
_user_specified_name	2999094:'#
!
_user_specified_name	2999092:'#
!
_user_specified_name	2999090:'#
!
_user_specified_name	2999088:'#
!
_user_specified_name	2999086:'#
!
_user_specified_name	2999084:'#
!
_user_specified_name	2999082:'#
!
_user_specified_name	2999080:'#
!
_user_specified_name	2999078:'#
!
_user_specified_name	2999076:'#
!
_user_specified_name	2999074:'#
!
_user_specified_name	2999072:'#
!
_user_specified_name	2999070:'#
!
_user_specified_name	2999068:'#
!
_user_specified_name	2999066:'#
!
_user_specified_name	2999064:'#
!
_user_specified_name	2999062:'#
!
_user_specified_name	2999060:'
#
!
_user_specified_name	2999058:'	#
!
_user_specified_name	2999056:'#
!
_user_specified_name	2999054:'#
!
_user_specified_name	2999052:'#
!
_user_specified_name	2999050:'#
!
_user_specified_name	2999048:_[
4
_output_shapes"
 :                  
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:         
!
_user_specified_name	t_input:P L
'
_output_shapes
:         
!
_user_specified_name	x_input
╫
√
D__inference_dense_2_layer_call_and_return_conditional_losses_2998627

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
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
:         @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:Г
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*+
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0|
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @T
TanhTanhBiasAdd:output:0*
T0*+
_output_shapes
:         @[
IdentityIdentityTanh:y:0^NoOp*
T0*+
_output_shapes
:         @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
╧
l
B__inference_add_1_layer_call_and_return_conditional_losses_2998595

inputs
inputs_1
identityT
addAddV2inputsinputs_1*
T0*+
_output_shapes
:         @S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:SO
+
_output_shapes
:         @
 
_user_specified_nameinputs:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ё	
щ
8__inference_spatial_transformation_layer_call_fn_2997849
spatial_layer1_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
identityИвStatefulPartitionedCallЧ
StatefulPartitionedCallStatefulPartitionedCallspatial_layer1_inputunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997823s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:         : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	2997845:'#
!
_user_specified_name	2997843:'#
!
_user_specified_name	2997841:'#
!
_user_specified_name	2997839:a ]
+
_output_shapes
:         
.
_user_specified_namespatial_layer1_input
╢
H
,__inference_activation_layer_call_fn_3000393

inputs
identity╣
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_activation_layer_call_and_return_conditional_losses_2997763d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
├
e
I__inference_activation_6_layer_call_and_return_conditional_losses_2998211

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:         @P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
─
■
G__inference_bpe_layer2_layer_call_and_return_conditional_losses_2997950

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 20
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
 :                  @
 
_user_specified_nameinputs
╙
e
I__inference_activation_1_layer_call_and_return_conditional_losses_3000447

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:         @T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
Х
├
@__inference_BPE_layer_call_and_return_conditional_losses_2997963
bpe_layer1_input$
bpe_layer1_2997910:@ 
bpe_layer1_2997912:@$
bpe_layer2_2997951:@@ 
bpe_layer2_2997953:@
identityИв"bpe_layer1/StatefulPartitionedCallв"bpe_layer2/StatefulPartitionedCallХ
"bpe_layer1/StatefulPartitionedCallStatefulPartitionedCallbpe_layer1_inputbpe_layer1_2997910bpe_layer1_2997912*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bpe_layer1_layer_call_and_return_conditional_losses_2997909Ў
activation_2/PartitionedCallPartitionedCall+bpe_layer1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_2997919к
"bpe_layer2/StatefulPartitionedCallStatefulPartitionedCall%activation_2/PartitionedCall:output:0bpe_layer2_2997951bpe_layer2_2997953*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bpe_layer2_layer_call_and_return_conditional_losses_2997950Ў
activation_3/PartitionedCallPartitionedCall+bpe_layer2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_3_layer_call_and_return_conditional_losses_2997960Б
IdentityIdentity%activation_3/PartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @l
NoOpNoOp#^bpe_layer1/StatefulPartitionedCall#^bpe_layer2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*;
_input_shapes*
(:                  : : : : 2H
"bpe_layer1/StatefulPartitionedCall"bpe_layer1/StatefulPartitionedCall2H
"bpe_layer2/StatefulPartitionedCall"bpe_layer2/StatefulPartitionedCall:'#
!
_user_specified_name	2997953:'#
!
_user_specified_name	2997951:'#
!
_user_specified_name	2997912:'#
!
_user_specified_name	2997910:f b
4
_output_shapes"
 :                  
*
_user_specified_namebpe_layer1_input
╣
S
'__inference_add_1_layer_call_fn_3000063
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_2998595d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:UQ
+
_output_shapes
:         @
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:         @
"
_user_specified_name
inputs_0
Й
d
H__inference_rescaling_2_layer_call_and_return_conditional_losses_2999685

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
 *  А┐b
mulMulinputsCast/x:output:0*
T0*4
_output_shapes"
 :                  g
addAddV2mul:z:0Cast_1/x:output:0*
T0*4
_output_shapes"
 :                  \
IdentityIdentityadd:z:0*
T0*4
_output_shapes"
 :                  "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  :\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
н
Щ
,__inference_bve_layer1_layer_call_fn_3000554

inputs
unknown:@
	unknown_0:@
identityИвStatefulPartitionedCallь
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *P
fKRI
G__inference_bve_layer1_layer_call_and_return_conditional_losses_2998065|
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*4
_output_shapes"
 :                  @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000550:'#
!
_user_specified_name	3000548:\ X
4
_output_shapes"
 :                  
 
_user_specified_nameinputs
┬
Ї
6__inference_multi_head_attention_layer_call_fn_2999752	
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
identityИвStatefulPartitionedCall╚
StatefulPartitionedCallStatefulPartitionedCallqueryvaluekeyunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2998440s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'
#
!
_user_specified_name	2999748:'	#
!
_user_specified_name	2999746:'#
!
_user_specified_name	2999744:'#
!
_user_specified_name	2999742:'#
!
_user_specified_name	2999740:'#
!
_user_specified_name	2999738:'#
!
_user_specified_name	2999736:'#
!
_user_specified_name	2999734:YU
4
_output_shapes"
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
к
J
.__inference_activation_6_layer_call_fn_3000667

inputs
identity╖
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_2998211`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
┐.
Е
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_3000057	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
є	
ё
@__inference_ou2_layer_call_and_return_conditional_losses_3000691

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
ў
e
I__inference_activation_5_layer_call_and_return_conditional_losses_3000643

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :                  @]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╫
n
B__inference_add_2_layer_call_and_return_conditional_losses_3000279
inputs_0
inputs_1
identityV
addAddV2inputs_0inputs_1*
T0*+
_output_shapes
:         @S
IdentityIdentityadd:z:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:UQ
+
_output_shapes
:         @
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:         @
"
_user_specified_name
inputs_0
╙
e
I__inference_activation_1_layer_call_and_return_conditional_losses_2997804

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:         @T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
мЛ
у
B__inference_model_layer_call_and_return_conditional_losses_2999041
x_input
t_input
	xbc_layer
	tbc_layer
	ubc_layer
bpe_2998811:@
bpe_2998813:@
bpe_2998815:@@
bpe_2998817:@
bve_2998820:@
bve_2998822:@
bve_2998824:@@
bve_2998826:@0
spatial_transformation_2998829:@,
spatial_transformation_2998831:@0
spatial_transformation_2998833:@@,
spatial_transformation_2998835:@2
multi_head_attention_2998874:@@.
multi_head_attention_2998876:@2
multi_head_attention_2998878:@@.
multi_head_attention_2998880:@2
multi_head_attention_2998882:@@.
multi_head_attention_2998884:@2
multi_head_attention_2998886:@@*
multi_head_attention_2998888:@
dense_2998892:@@
dense_2998894:@!
dense_1_2998897:@@
dense_1_2998899:@4
multi_head_attention_1_2998938:@@0
multi_head_attention_1_2998940:@4
multi_head_attention_1_2998942:@@0
multi_head_attention_1_2998944:@4
multi_head_attention_1_2998946:@@0
multi_head_attention_1_2998948:@4
multi_head_attention_1_2998950:@@,
multi_head_attention_1_2998952:@!
dense_2_2998956:@@
dense_2_2998958:@!
dense_3_2998961:@@
dense_3_2998963:@4
multi_head_attention_2_2999002:@@0
multi_head_attention_2_2999004:@4
multi_head_attention_2_2999006:@@0
multi_head_attention_2_2999008:@4
multi_head_attention_2_2999010:@@0
multi_head_attention_2_2999012:@4
multi_head_attention_2_2999014:@@,
multi_head_attention_2_2999016:@!
dense_4_2999020:@@
dense_4_2999022:@
	u_2999026:@@
	u_2999028:@
	u_2999030:@@
	u_2999032:@"
output_u_2999035:@
output_u_2999037:
identityИвBPE/StatefulPartitionedCallвBVE/StatefulPartitionedCallвU/StatefulPartitionedCallвdense/StatefulPartitionedCallвdense_1/StatefulPartitionedCallвdense_2/StatefulPartitionedCallвdense_3/StatefulPartitionedCallвdense_4/StatefulPartitionedCallв,multi_head_attention/StatefulPartitionedCallв.multi_head_attention_1/StatefulPartitionedCallв.multi_head_attention_2/StatefulPartitionedCallв output_u/StatefulPartitionedCallв.spatial_transformation/StatefulPartitionedCall┐
rescaling/PartitionedCallPartitionedCallx_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *O
fJRH
F__inference_rescaling_layer_call_and_return_conditional_losses_2998319├
rescaling_1/PartitionedCallPartitionedCallt_input*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_1_layer_call_and_return_conditional_losses_2998328╥
rescaling_2/PartitionedCallPartitionedCall	xbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_2_layer_call_and_return_conditional_losses_2998337╥
rescaling_3/PartitionedCallPartitionedCall	tbc_layer*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_rescaling_3_layer_call_and_return_conditional_losses_2998346Е
concatenate/PartitionedCallPartitionedCall"rescaling/PartitionedCall:output:0$rescaling_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_2998354Ш
concatenate_1/PartitionedCallPartitionedCall$rescaling_2/PartitionedCall:output:0$rescaling_3/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *S
fNRL
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2998362▄
reshape/PartitionedCallPartitionedCall$concatenate/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_reshape_layer_call_and_return_conditional_losses_2998376н
BPE/StatefulPartitionedCallStatefulPartitionedCall&concatenate_1/PartitionedCall:output:0bpe_2998811bpe_2998813bpe_2998815bpe_2998817*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_BPE_layer_call_and_return_conditional_losses_2997979Р
BVE/StatefulPartitionedCallStatefulPartitionedCall	ubc_layerbve_2998820bve_2998822bve_2998824bve_2998826*
Tin	
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_BVE_layer_call_and_return_conditional_losses_2998135Р
.spatial_transformation/StatefulPartitionedCallStatefulPartitionedCall reshape/PartitionedCall:output:0spatial_transformation_2998829spatial_transformation_2998831spatial_transformation_2998833spatial_transformation_2998835*
Tin	
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997823щ
,multi_head_attention/StatefulPartitionedCallStatefulPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:0$BVE/StatefulPartitionedCall:output:0$BPE/StatefulPartitionedCall:output:0multi_head_attention_2998874multi_head_attention_2998876multi_head_attention_2998878multi_head_attention_2998880multi_head_attention_2998882multi_head_attention_2998884multi_head_attention_2998886multi_head_attention_2998888*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *Z
fURS
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2998873Я
add/PartitionedCallPartitionedCall7spatial_transformation/StatefulPartitionedCall:output:05multi_head_attention/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_add_layer_call_and_return_conditional_losses_2998463Д
dense/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0dense_2998892dense_2998894*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_dense_layer_call_and_return_conditional_losses_2998495Ц
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_2998897dense_1_2998899*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_1_layer_call_and_return_conditional_losses_2998531ю
.multi_head_attention_1/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0$BVE/StatefulPartitionedCall:output:0$BPE/StatefulPartitionedCall:output:0multi_head_attention_1_2998938multi_head_attention_1_2998940multi_head_attention_1_2998942multi_head_attention_1_2998944multi_head_attention_1_2998946multi_head_attention_1_2998948multi_head_attention_1_2998950multi_head_attention_1_2998952*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_2998937Ц
add_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:07multi_head_attention_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_add_1_layer_call_and_return_conditional_losses_2998595О
dense_2/StatefulPartitionedCallStatefulPartitionedCalladd_1/PartitionedCall:output:0dense_2_2998956dense_2_2998958*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_2_layer_call_and_return_conditional_losses_2998627Ш
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_2998961dense_3_2998963*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_2998663ю
.multi_head_attention_2/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0$BVE/StatefulPartitionedCall:output:0$BPE/StatefulPartitionedCall:output:0multi_head_attention_2_2999002multi_head_attention_2_2999004multi_head_attention_2_2999006multi_head_attention_2_2999008multi_head_attention_2_2999010multi_head_attention_2_2999012multi_head_attention_2_2999014multi_head_attention_2_2999016*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @**
_read_only_resource_inputs

	
*0
config_proto 

CPU

GPU2*0J 8В *\
fWRU
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_2999001Ц
add_2/PartitionedCallPartitionedCall(dense_3/StatefulPartitionedCall:output:07multi_head_attention_2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_2998727О
dense_4/StatefulPartitionedCallStatefulPartitionedCalladd_2/PartitionedCall:output:0dense_4_2999020dense_4_2999022*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_4_layer_call_and_return_conditional_losses_2998759▄
flatten/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_flatten_layer_call_and_return_conditional_losses_2998770О
U/StatefulPartitionedCallStatefulPartitionedCall flatten/PartitionedCall:output:0	u_2999026	u_2999028	u_2999030	u_2999032*
Tin	
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*&
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *G
fBR@
>__inference_U_layer_call_and_return_conditional_losses_2998251Т
 output_u/StatefulPartitionedCallStatefulPartitionedCall"U/StatefulPartitionedCall:output:0output_u_2999035output_u_2999037*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *N
fIRG
E__inference_output_u_layer_call_and_return_conditional_losses_2998790x
IdentityIdentity)output_u/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         З
NoOpNoOp^BPE/StatefulPartitionedCall^BVE/StatefulPartitionedCall^U/StatefulPartitionedCall^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall-^multi_head_attention/StatefulPartitionedCall/^multi_head_attention_1/StatefulPartitionedCall/^multi_head_attention_2/StatefulPartitionedCall!^output_u/StatefulPartitionedCall/^spatial_transformation/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesё
ю:         :         :                  :                  :                  : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2:
BPE/StatefulPartitionedCallBPE/StatefulPartitionedCall2:
BVE/StatefulPartitionedCallBVE/StatefulPartitionedCall26
U/StatefulPartitionedCallU/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2\
,multi_head_attention/StatefulPartitionedCall,multi_head_attention/StatefulPartitionedCall2`
.multi_head_attention_1/StatefulPartitionedCall.multi_head_attention_1/StatefulPartitionedCall2`
.multi_head_attention_2/StatefulPartitionedCall.multi_head_attention_2/StatefulPartitionedCall2D
 output_u/StatefulPartitionedCall output_u/StatefulPartitionedCall2`
.spatial_transformation/StatefulPartitionedCall.spatial_transformation/StatefulPartitionedCall:'8#
!
_user_specified_name	2999037:'7#
!
_user_specified_name	2999035:'6#
!
_user_specified_name	2999032:'5#
!
_user_specified_name	2999030:'4#
!
_user_specified_name	2999028:'3#
!
_user_specified_name	2999026:'2#
!
_user_specified_name	2999022:'1#
!
_user_specified_name	2999020:'0#
!
_user_specified_name	2999016:'/#
!
_user_specified_name	2999014:'.#
!
_user_specified_name	2999012:'-#
!
_user_specified_name	2999010:',#
!
_user_specified_name	2999008:'+#
!
_user_specified_name	2999006:'*#
!
_user_specified_name	2999004:')#
!
_user_specified_name	2999002:'(#
!
_user_specified_name	2998963:''#
!
_user_specified_name	2998961:'&#
!
_user_specified_name	2998958:'%#
!
_user_specified_name	2998956:'$#
!
_user_specified_name	2998952:'##
!
_user_specified_name	2998950:'"#
!
_user_specified_name	2998948:'!#
!
_user_specified_name	2998946:' #
!
_user_specified_name	2998944:'#
!
_user_specified_name	2998942:'#
!
_user_specified_name	2998940:'#
!
_user_specified_name	2998938:'#
!
_user_specified_name	2998899:'#
!
_user_specified_name	2998897:'#
!
_user_specified_name	2998894:'#
!
_user_specified_name	2998892:'#
!
_user_specified_name	2998888:'#
!
_user_specified_name	2998886:'#
!
_user_specified_name	2998884:'#
!
_user_specified_name	2998882:'#
!
_user_specified_name	2998880:'#
!
_user_specified_name	2998878:'#
!
_user_specified_name	2998876:'#
!
_user_specified_name	2998874:'#
!
_user_specified_name	2998835:'#
!
_user_specified_name	2998833:'#
!
_user_specified_name	2998831:'#
!
_user_specified_name	2998829:'#
!
_user_specified_name	2998826:'#
!
_user_specified_name	2998824:'
#
!
_user_specified_name	2998822:'	#
!
_user_specified_name	2998820:'#
!
_user_specified_name	2998817:'#
!
_user_specified_name	2998815:'#
!
_user_specified_name	2998813:'#
!
_user_specified_name	2998811:_[
4
_output_shapes"
 :                  
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	tbc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	Xbc_layer:PL
'
_output_shapes
:         
!
_user_specified_name	t_input:P L
'
_output_shapes
:         
!
_user_specified_name	x_input
╞
b
F__inference_rescaling_layer_call_and_return_conditional_losses_2999646

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
 *  А┐U
mulMulinputsCast/x:output:0*
T0*'
_output_shapes
:         Z
addAddV2mul:z:0Cast_1/x:output:0*
T0*'
_output_shapes
:         O
IdentityIdentityadd:z:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         :O K
'
_output_shapes
:         
 
_user_specified_nameinputs
├
e
I__inference_activation_7_layer_call_and_return_conditional_losses_3000701

inputs
identityF
TanhTanhinputs*
T0*'
_output_shapes
:         @P
IdentityIdentityTanh:y:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs
╜.
Г
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2999811	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
╣
S
'__inference_add_2_layer_call_fn_3000273
inputs_0
inputs_1
identity┴
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *K
fFRD
B__inference_add_2_layer_call_and_return_conditional_losses_2998727d
IdentityIdentityPartitionedCall:output:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:         @:         @:UQ
+
_output_shapes
:         @
"
_user_specified_name
inputs_1:U Q
+
_output_shapes
:         @
"
_user_specified_name
inputs_0
С
Э
0__inference_spatial_layer2_layer_call_fn_3000407

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallч
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *T
fORM
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_2997794s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000403:'#
!
_user_specified_name	3000401:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
─
■
G__inference_bve_layer2_layer_call_and_return_conditional_losses_3000633

inputs3
!tensordot_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  @К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  @: : 20
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
 :                  @
 
_user_specified_nameinputs
┐.
Е
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_2998572	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
 -
Р
%__inference_signature_wrapper_2999633
	xbc_layer
t_input
	tbc_layer
	ubc_layer
x_input
unknown:@
	unknown_0:@
	unknown_1:@@
	unknown_2:@
	unknown_3:@
	unknown_4:@
	unknown_5:@@
	unknown_6:@
	unknown_7:@
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

unknown_23:@@

unknown_24:@ 

unknown_25:@@

unknown_26:@ 

unknown_27:@@

unknown_28:@ 

unknown_29:@@

unknown_30:@

unknown_31:@@

unknown_32:@

unknown_33:@@

unknown_34:@ 

unknown_35:@@

unknown_36:@ 

unknown_37:@@

unknown_38:@ 

unknown_39:@@

unknown_40:@ 

unknown_41:@@

unknown_42:@

unknown_43:@@

unknown_44:@

unknown_45:@@

unknown_46:@

unknown_47:@@

unknown_48:@

unknown_49:@

unknown_50:
identityИвStatefulPartitionedCallЬ
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
unknown_38
unknown_39
unknown_40
unknown_41
unknown_42
unknown_43
unknown_44
unknown_45
unknown_46
unknown_47
unknown_48
unknown_49
unknown_50*D
Tin=
;29*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         *V
_read_only_resource_inputs8
64	
 !"#$%&'()*+,-./012345678*0
config_proto 

CPU

GPU2*0J 8В *+
f&R$
"__inference__wrapped_model_2997721o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         <
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*Г
_input_shapesё
ю:                  :         :                  :                  :         : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:'8#
!
_user_specified_name	2999629:'7#
!
_user_specified_name	2999627:'6#
!
_user_specified_name	2999625:'5#
!
_user_specified_name	2999623:'4#
!
_user_specified_name	2999621:'3#
!
_user_specified_name	2999619:'2#
!
_user_specified_name	2999617:'1#
!
_user_specified_name	2999615:'0#
!
_user_specified_name	2999613:'/#
!
_user_specified_name	2999611:'.#
!
_user_specified_name	2999609:'-#
!
_user_specified_name	2999607:',#
!
_user_specified_name	2999605:'+#
!
_user_specified_name	2999603:'*#
!
_user_specified_name	2999601:')#
!
_user_specified_name	2999599:'(#
!
_user_specified_name	2999597:''#
!
_user_specified_name	2999595:'&#
!
_user_specified_name	2999593:'%#
!
_user_specified_name	2999591:'$#
!
_user_specified_name	2999589:'##
!
_user_specified_name	2999587:'"#
!
_user_specified_name	2999585:'!#
!
_user_specified_name	2999583:' #
!
_user_specified_name	2999581:'#
!
_user_specified_name	2999579:'#
!
_user_specified_name	2999577:'#
!
_user_specified_name	2999575:'#
!
_user_specified_name	2999573:'#
!
_user_specified_name	2999571:'#
!
_user_specified_name	2999569:'#
!
_user_specified_name	2999567:'#
!
_user_specified_name	2999565:'#
!
_user_specified_name	2999563:'#
!
_user_specified_name	2999561:'#
!
_user_specified_name	2999559:'#
!
_user_specified_name	2999557:'#
!
_user_specified_name	2999555:'#
!
_user_specified_name	2999553:'#
!
_user_specified_name	2999551:'#
!
_user_specified_name	2999549:'#
!
_user_specified_name	2999547:'#
!
_user_specified_name	2999545:'#
!
_user_specified_name	2999543:'#
!
_user_specified_name	2999541:'#
!
_user_specified_name	2999539:'
#
!
_user_specified_name	2999537:'	#
!
_user_specified_name	2999535:'#
!
_user_specified_name	2999533:'#
!
_user_specified_name	2999531:'#
!
_user_specified_name	2999529:'#
!
_user_specified_name	2999527:PL
'
_output_shapes
:         
!
_user_specified_name	x_input:_[
4
_output_shapes"
 :                  
#
_user_specified_name	ubc_layer:_[
4
_output_shapes"
 :                  
#
_user_specified_name	tbc_layer:PL
'
_output_shapes
:         
!
_user_specified_name	t_input:_ [
4
_output_shapes"
 :                  
#
_user_specified_name	Xbc_layer
▐
J
.__inference_activation_2_layer_call_fn_3000491

inputs
identity─
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *4
_output_shapes"
 :                  @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_2_layer_call_and_return_conditional_losses_2997919m
IdentityIdentityPartitionedCall:output:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
ў
e
I__inference_activation_2_layer_call_and_return_conditional_losses_3000496

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :                  @]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
Г
Ц
)__inference_dense_3_layer_call_fn_3000118

inputs
unknown:@@
	unknown_0:@
identityИвStatefulPartitionedCallр
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *M
fHRF
D__inference_dense_3_layer_call_and_return_conditional_losses_2998663s
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*+
_output_shapes
:         @<
NoOpNoOp^StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : 22
StatefulPartitionedCallStatefulPartitionedCall:'#
!
_user_specified_name	3000114:'#
!
_user_specified_name	3000112:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
╝
`
D__inference_flatten_layer_call_and_return_conditional_losses_2998770

inputs
identityV
ConstConst*
_output_shapes
:*
dtype0*
valueB"    @   \
ReshapeReshapeinputsConst:output:0*
T0*'
_output_shapes
:         @X
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ў
e
I__inference_activation_4_layer_call_and_return_conditional_losses_3000594

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :                  @]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
╤
c
G__inference_activation_layer_call_and_return_conditional_losses_2997763

inputs
identityJ
TanhTanhinputs*
T0*+
_output_shapes
:         @T
IdentityIdentityTanh:y:0*
T0*+
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @:S O
+
_output_shapes
:         @
 
_user_specified_nameinputs
ў
e
I__inference_activation_3_layer_call_and_return_conditional_losses_3000545

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :                  @]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
─
■
G__inference_bve_layer1_layer_call_and_return_conditional_losses_3000584

inputs3
!tensordot_readvariableop_resource:@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвTensordot/ReadVariableOpz
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
::э╧Y
Tensordot/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : ╗
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
value	B : ┐
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
value	B : Ь
Tensordot/concatConcatV2Tensordot/free:output:0Tensordot/axes:output:0Tensordot/concat/axis:output:0*
N*
T0*
_output_shapes
:y
Tensordot/stackPackTensordot/Prod:output:0Tensordot/Prod_1:output:0*
N*
T0*
_output_shapes
:В
Tensordot/transpose	TransposeinputsTensordot/concat:output:0*
T0*4
_output_shapes"
 :                  К
Tensordot/ReshapeReshapeTensordot/transpose:y:0Tensordot/stack:output:0*
T0*0
_output_shapes
:                  К
Tensordot/MatMulMatMulTensordot/Reshape:output:0 Tensordot/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @[
Tensordot/Const_2Const*
_output_shapes
:*
dtype0*
valueB:@Y
Tensordot/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : з
Tensordot/concat_1ConcatV2Tensordot/GatherV2:output:0Tensordot/Const_2:output:0 Tensordot/concat_1/axis:output:0*
N*
T0*
_output_shapes
:М
	TensordotReshapeTensordot/MatMul:product:0Tensordot/concat_1:output:0*
T0*4
_output_shapes"
 :                  @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0Е
BiasAddBiasAddTensordot:output:0BiasAdd/ReadVariableOp:value:0*
T0*4
_output_shapes"
 :                  @l
IdentityIdentityBiasAdd:output:0^NoOp*
T0*4
_output_shapes"
 :                  @V
NoOpNoOp^BiasAdd/ReadVariableOp^Tensordot/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:                  : : 20
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
 :                  
 
_user_specified_nameinputs
ў
e
I__inference_activation_3_layer_call_and_return_conditional_losses_2997960

inputs
identityS
TanhTanhinputs*
T0*4
_output_shapes"
 :                  @]
IdentityIdentityTanh:y:0*
T0*4
_output_shapes"
 :                  @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :                  @:\ X
4
_output_shapes"
 :                  @
 
_user_specified_nameinputs
И
Р
>__inference_U_layer_call_and_return_conditional_losses_2998235
	ou1_input
ou1_2998202:@@
ou1_2998204:@
ou2_2998223:@@
ou2_2998225:@
identityИвou1/StatefulPartitionedCallвou2/StatefulPartitionedCallх
ou1/StatefulPartitionedCallStatefulPartitionedCall	ou1_inputou1_2998202ou1_2998204*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_ou1_layer_call_and_return_conditional_losses_2998201т
activation_6/PartitionedCallPartitionedCall$ou1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_6_layer_call_and_return_conditional_losses_2998211Б
ou2/StatefulPartitionedCallStatefulPartitionedCall%activation_6/PartitionedCall:output:0ou2_2998223ou2_2998225*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8В *I
fDRB
@__inference_ou2_layer_call_and_return_conditional_losses_2998222т
activation_7/PartitionedCallPartitionedCall$ou2/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_2998232t
IdentityIdentity%activation_7/PartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:         @^
NoOpNoOp^ou1/StatefulPartitionedCall^ou2/StatefulPartitionedCall*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:         @: : : : 2:
ou1/StatefulPartitionedCallou1/StatefulPartitionedCall2:
ou2/StatefulPartitionedCallou2/StatefulPartitionedCall:'#
!
_user_specified_name	2998225:'#
!
_user_specified_name	2998223:'#
!
_user_specified_name	2998204:'#
!
_user_specified_name	2998202:R N
'
_output_shapes
:         @
#
_user_specified_name	ou1_input
┐.
Е
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_3000267	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
н
Y
-__inference_concatenate_layer_call_fn_2999665
inputs_0
inputs_1
identity├
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *Q
fLRJ
H__inference_concatenate_layer_call_and_return_conditional_losses_2998354`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*9
_input_shapes(
&:         :         :QM
'
_output_shapes
:         
"
_user_specified_name
inputs_1:Q M
'
_output_shapes
:         
"
_user_specified_name
inputs_0
╜.
Г
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2998873	
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
identityИв#attention_output/add/ReadVariableOpв-attention_output/einsum/Einsum/ReadVariableOpвkey/add/ReadVariableOpв key/einsum/Einsum/ReadVariableOpвquery/add/ReadVariableOpв"query/einsum/Einsum/ReadVariableOpвvalue/add/ReadVariableOpв"value/einsum/Einsum/ReadVariableOpТ
"query/einsum/Einsum/ReadVariableOpReadVariableOp+query_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0░
query/einsum/EinsumEinsumquery*query/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*/
_output_shapes
:         @*
equationabc,cde->abdez
query/add/ReadVariableOpReadVariableOp!query_add_readvariableop_resource*
_output_shapes

:@*
dtype0Р
query/add/addAddV2query/einsum/Einsum:output:0 query/add/ReadVariableOp:value:0*
T0*/
_output_shapes
:         @О
 key/einsum/Einsum/ReadVariableOpReadVariableOp)key_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0│
key/einsum/EinsumEinsumkey(key/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdev
key/add/ReadVariableOpReadVariableOpkey_add_readvariableop_resource*
_output_shapes

:@*
dtype0У
key/add/addAddV2key/einsum/Einsum:output:0key/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @Т
"value/einsum/Einsum/ReadVariableOpReadVariableOp+value_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╣
value/einsum/EinsumEinsumvalue*value/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*8
_output_shapes&
$:"                  @*
equationabc,cde->abdez
value/add/ReadVariableOpReadVariableOp!value_add_readvariableop_resource*
_output_shapes

:@*
dtype0Щ
value/add/addAddV2value/einsum/Einsum:output:0 value/add/ReadVariableOp:value:0*
T0*8
_output_shapes&
$:"                  @J
Mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *   >g
MulMulquery/add/add:z:0Mul/y:output:0*
T0*/
_output_shapes
:         @Ь
einsum/EinsumEinsumkey/add/add:z:0Mul:z:0*
N*
T0*8
_output_shapes&
$:"                  *
equationaecd,abcd->acbeu
softmax/SoftmaxSoftmaxeinsum/Einsum:output:0*
T0*8
_output_shapes&
$:"                  z
dropout/IdentityIdentitysoftmax/Softmax:softmax:0*
T0*8
_output_shapes&
$:"                  й
einsum_1/EinsumEinsumdropout/Identity:output:0value/add/add:z:0*
N*
T0*/
_output_shapes
:         @*
equationacbe,aecd->abcdи
-attention_output/einsum/Einsum/ReadVariableOpReadVariableOp6attention_output_einsum_einsum_readvariableop_resource*"
_output_shapes
:@@*
dtype0╒
attention_output/einsum/EinsumEinsumeinsum_1/Einsum:output:05attention_output/einsum/Einsum/ReadVariableOp:value:0*
N*
T0*+
_output_shapes
:         @*
equationabcd,cde->abeМ
#attention_output/add/ReadVariableOpReadVariableOp,attention_output_add_readvariableop_resource*
_output_shapes
:@*
dtype0н
attention_output/add/addAddV2'attention_output/einsum/Einsum:output:0+attention_output/add/ReadVariableOp:value:0*
T0*+
_output_shapes
:         @o
IdentityIdentityattention_output/add/add:z:0^NoOp*
T0*+
_output_shapes
:         @┤
NoOpNoOp$^attention_output/add/ReadVariableOp.^attention_output/einsum/Einsum/ReadVariableOp^key/add/ReadVariableOp!^key/einsum/Einsum/ReadVariableOp^query/add/ReadVariableOp#^query/einsum/Einsum/ReadVariableOp^value/add/ReadVariableOp#^value/einsum/Einsum/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*z
_input_shapesi
g:         @:                  @:                  @: : : : : : : : 2J
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
 :                  @

_user_specified_namekey:[W
4
_output_shapes"
 :                  @

_user_specified_namevalue:R N
+
_output_shapes
:         @

_user_specified_namequery
є	
ё
@__inference_ou1_layer_call_and_return_conditional_losses_3000662

inputs0
matmul_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identityИвBiasAdd/ReadVariableOpвMatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@@*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:         @_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:         @S
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:         @: : 20
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
:         @
 
_user_specified_nameinputs
к
J
.__inference_activation_7_layer_call_fn_3000696

inputs
identity╖
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:         @* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8В *R
fMRK
I__inference_activation_7_layer_call_and_return_conditional_losses_2998232`
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:         @"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*&
_input_shapes
:         @:O K
'
_output_shapes
:         @
 
_user_specified_nameinputs"эL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*╥
serving_default╛
L
	Xbc_layer?
serving_default_Xbc_layer:0                  
;
t_input0
serving_default_t_input:0         
L
	tbc_layer?
serving_default_tbc_layer:0                  
L
	ubc_layer?
serving_default_ubc_layer:0                  
;
x_input0
serving_default_x_input:0         <
output_u0
StatefulPartitionedCall:0         tensorflow/serving/predict:Оя
╡
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
layer-16
layer_with_weights-4
layer-17
layer_with_weights-5
layer-18
layer_with_weights-6
layer-19
layer-20
layer_with_weights-7
layer-21
layer_with_weights-8
layer-22
layer_with_weights-9
layer-23
layer-24
layer_with_weights-10
layer-25
layer-26
layer_with_weights-11
layer-27
layer_with_weights-12
layer-28
	variables
trainable_variables
 regularization_losses
!	keras_api
"__call__
*#&call_and_return_all_conditional_losses
$_default_save_signature
%
signatures"
_tf_keras_network
"
_tf_keras_input_layer
"
_tf_keras_input_layer
е
&	variables
'trainable_variables
(regularization_losses
)	keras_api
*__call__
*+&call_and_return_all_conditional_losses"
_tf_keras_layer
е
,	variables
-trainable_variables
.regularization_losses
/	keras_api
0__call__
*1&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
"
_tf_keras_input_layer
е
2	variables
3trainable_variables
4regularization_losses
5	keras_api
6__call__
*7&call_and_return_all_conditional_losses"
_tf_keras_layer
е
8	variables
9trainable_variables
:regularization_losses
;	keras_api
<__call__
*=&call_and_return_all_conditional_losses"
_tf_keras_layer
е
>	variables
?trainable_variables
@regularization_losses
A	keras_api
B__call__
*C&call_and_return_all_conditional_losses"
_tf_keras_layer
е
D	variables
Etrainable_variables
Fregularization_losses
G	keras_api
H__call__
*I&call_and_return_all_conditional_losses"
_tf_keras_layer
е
J	variables
Ktrainable_variables
Lregularization_losses
M	keras_api
N__call__
*O&call_and_return_all_conditional_losses"
_tf_keras_layer
"
_tf_keras_input_layer
Т
Player_with_weights-0
Player-0
Qlayer-1
Rlayer_with_weights-1
Rlayer-2
Slayer-3
T	variables
Utrainable_variables
Vregularization_losses
W	keras_api
X__call__
*Y&call_and_return_all_conditional_losses"
_tf_keras_sequential
Т
Zlayer_with_weights-0
Zlayer-0
[layer-1
\layer_with_weights-1
\layer-2
]layer-3
^	variables
_trainable_variables
`regularization_losses
a	keras_api
b__call__
*c&call_and_return_all_conditional_losses"
_tf_keras_sequential
Т
dlayer_with_weights-0
dlayer-0
elayer-1
flayer_with_weights-1
flayer-2
glayer-3
h	variables
itrainable_variables
jregularization_losses
k	keras_api
l__call__
*m&call_and_return_all_conditional_losses"
_tf_keras_sequential
О
n	variables
otrainable_variables
pregularization_losses
q	keras_api
r__call__
*s&call_and_return_all_conditional_losses
t_query_dense
u
_key_dense
v_value_dense
w_softmax
x_dropout_layer
y_output_dense"
_tf_keras_layer
е
z	variables
{trainable_variables
|regularization_losses
}	keras_api
~__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
├
А	variables
Бtrainable_variables
Вregularization_losses
Г	keras_api
Д__call__
+Е&call_and_return_all_conditional_losses
Жkernel
	Зbias"
_tf_keras_layer
├
И	variables
Йtrainable_variables
Кregularization_losses
Л	keras_api
М__call__
+Н&call_and_return_all_conditional_losses
Оkernel
	Пbias"
_tf_keras_layer
Ъ
Р	variables
Сtrainable_variables
Тregularization_losses
У	keras_api
Ф__call__
+Х&call_and_return_all_conditional_losses
Ц_query_dense
Ч
_key_dense
Ш_value_dense
Щ_softmax
Ъ_dropout_layer
Ы_output_dense"
_tf_keras_layer
л
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses"
_tf_keras_layer
├
в	variables
гtrainable_variables
дregularization_losses
е	keras_api
ж__call__
+з&call_and_return_all_conditional_losses
иkernel
	йbias"
_tf_keras_layer
├
к	variables
лtrainable_variables
мregularization_losses
н	keras_api
о__call__
+п&call_and_return_all_conditional_losses
░kernel
	▒bias"
_tf_keras_layer
Ъ
▓	variables
│trainable_variables
┤regularization_losses
╡	keras_api
╢__call__
+╖&call_and_return_all_conditional_losses
╕_query_dense
╣
_key_dense
║_value_dense
╗_softmax
╝_dropout_layer
╜_output_dense"
_tf_keras_layer
л
╛	variables
┐trainable_variables
└regularization_losses
┴	keras_api
┬__call__
+├&call_and_return_all_conditional_losses"
_tf_keras_layer
├
─	variables
┼trainable_variables
╞regularization_losses
╟	keras_api
╚__call__
+╔&call_and_return_all_conditional_losses
╩kernel
	╦bias"
_tf_keras_layer
л
╠	variables
═trainable_variables
╬regularization_losses
╧	keras_api
╨__call__
+╤&call_and_return_all_conditional_losses"
_tf_keras_layer
Ю
╥layer_with_weights-0
╥layer-0
╙layer-1
╘layer_with_weights-1
╘layer-2
╒layer-3
╓	variables
╫trainable_variables
╪regularization_losses
┘	keras_api
┌__call__
+█&call_and_return_all_conditional_losses"
_tf_keras_sequential
├
▄	variables
▌trainable_variables
▐regularization_losses
▀	keras_api
р__call__
+с&call_and_return_all_conditional_losses
тkernel
	уbias"
_tf_keras_layer
ъ
ф0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
Ё12
ё13
Є14
є15
Ї16
ї17
Ў18
ў19
Ж20
З21
О22
П23
°24
∙25
·26
√27
№28
¤29
■30
 31
и32
й33
░34
▒35
А36
Б37
В38
Г39
Д40
Е41
Ж42
З43
╩44
╦45
И46
Й47
К48
Л49
т50
у51"
trackable_list_wrapper
ъ
ф0
х1
ц2
ч3
ш4
щ5
ъ6
ы7
ь8
э9
ю10
я11
Ё12
ё13
Є14
є15
Ї16
ї17
Ў18
ў19
Ж20
З21
О22
П23
°24
∙25
·26
√27
№28
¤29
■30
 31
и32
й33
░34
▒35
А36
Б37
В38
Г39
Д40
Е41
Ж42
З43
╩44
╦45
И46
Й47
К48
Л49
т50
у51"
trackable_list_wrapper
 "
trackable_list_wrapper
╧
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
	variables
trainable_variables
 regularization_losses
"__call__
$_default_save_signature
*#&call_and_return_all_conditional_losses
&#"call_and_return_conditional_losses"
_generic_user_object
┼
Сtrace_0
Тtrace_12К
'__inference_model_layer_call_fn_2999154
'__inference_model_layer_call_fn_2999267╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zСtrace_0zТtrace_1
√
Уtrace_0
Фtrace_12└
B__inference_model_layer_call_and_return_conditional_losses_2998797
B__inference_model_layer_call_and_return_conditional_losses_2999041╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zУtrace_0zФtrace_1
ўBЇ
"__inference__wrapped_model_2997721x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"Ш
С▓Н
FullArgSpec
argsЪ

jargs_0
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
-
Хserving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
&	variables
'trainable_variables
(regularization_losses
*__call__
*+&call_and_return_all_conditional_losses
&+"call_and_return_conditional_losses"
_generic_user_object
ч
Ыtrace_02╚
+__inference_rescaling_layer_call_fn_2999638Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЫtrace_0
В
Ьtrace_02у
F__inference_rescaling_layer_call_and_return_conditional_losses_2999646Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЬtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Эnon_trainable_variables
Юlayers
Яmetrics
 аlayer_regularization_losses
бlayer_metrics
,	variables
-trainable_variables
.regularization_losses
0__call__
*1&call_and_return_all_conditional_losses
&1"call_and_return_conditional_losses"
_generic_user_object
щ
вtrace_02╩
-__inference_rescaling_1_layer_call_fn_2999651Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zвtrace_0
Д
гtrace_02х
H__inference_rescaling_1_layer_call_and_return_conditional_losses_2999659Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zгtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
дnon_trainable_variables
еlayers
жmetrics
 зlayer_regularization_losses
иlayer_metrics
2	variables
3trainable_variables
4regularization_losses
6__call__
*7&call_and_return_all_conditional_losses
&7"call_and_return_conditional_losses"
_generic_user_object
щ
йtrace_02╩
-__inference_concatenate_layer_call_fn_2999665Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zйtrace_0
Д
кtrace_02х
H__inference_concatenate_layer_call_and_return_conditional_losses_2999672Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zкtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
8	variables
9trainable_variables
:regularization_losses
<__call__
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
щ
░trace_02╩
-__inference_rescaling_2_layer_call_fn_2999677Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z░trace_0
Д
▒trace_02х
H__inference_rescaling_2_layer_call_and_return_conditional_losses_2999685Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z▒trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▓non_trainable_variables
│layers
┤metrics
 ╡layer_regularization_losses
╢layer_metrics
>	variables
?trainable_variables
@regularization_losses
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
щ
╖trace_02╩
-__inference_rescaling_3_layer_call_fn_2999690Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╖trace_0
Д
╕trace_02х
H__inference_rescaling_3_layer_call_and_return_conditional_losses_2999698Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╕trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
D	variables
Etrainable_variables
Fregularization_losses
H__call__
*I&call_and_return_all_conditional_losses
&I"call_and_return_conditional_losses"
_generic_user_object
х
╛trace_02╞
)__inference_reshape_layer_call_fn_2999703Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╛trace_0
А
┐trace_02с
D__inference_reshape_layer_call_and_return_conditional_losses_2999716Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z┐trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
J	variables
Ktrainable_variables
Lregularization_losses
N__call__
*O&call_and_return_all_conditional_losses
&O"call_and_return_conditional_losses"
_generic_user_object
ы
┼trace_02╠
/__inference_concatenate_1_layer_call_fn_2999722Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z┼trace_0
Ж
╞trace_02ч
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2999729Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╞trace_0
├
╟	variables
╚trainable_variables
╔regularization_losses
╩	keras_api
╦__call__
+╠&call_and_return_all_conditional_losses
фkernel
	хbias"
_tf_keras_layer
л
═	variables
╬trainable_variables
╧regularization_losses
╨	keras_api
╤__call__
+╥&call_and_return_all_conditional_losses"
_tf_keras_layer
├
╙	variables
╘trainable_variables
╒regularization_losses
╓	keras_api
╫__call__
+╪&call_and_return_all_conditional_losses
цkernel
	чbias"
_tf_keras_layer
л
┘	variables
┌trainable_variables
█regularization_losses
▄	keras_api
▌__call__
+▐&call_and_return_all_conditional_losses"
_tf_keras_layer
@
ф0
х1
ц2
ч3"
trackable_list_wrapper
@
ф0
х1
ц2
ч3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
▀non_trainable_variables
рlayers
сmetrics
 тlayer_regularization_losses
уlayer_metrics
T	variables
Utrainable_variables
Vregularization_losses
X__call__
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
ч
фtrace_0
хtrace_12м
8__inference_spatial_transformation_layer_call_fn_2997836
8__inference_spatial_transformation_layer_call_fn_2997849╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zфtrace_0zхtrace_1
Э
цtrace_0
чtrace_12т
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997807
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997823╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zцtrace_0zчtrace_1
├
ш	variables
щtrainable_variables
ъregularization_losses
ы	keras_api
ь__call__
+э&call_and_return_all_conditional_losses
шkernel
	щbias"
_tf_keras_layer
л
ю	variables
яtrainable_variables
Ёregularization_losses
ё	keras_api
Є__call__
+є&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Ї	variables
їtrainable_variables
Ўregularization_losses
ў	keras_api
°__call__
+∙&call_and_return_all_conditional_losses
ъkernel
	ыbias"
_tf_keras_layer
л
·	variables
√trainable_variables
№regularization_losses
¤	keras_api
■__call__
+ &call_and_return_all_conditional_losses"
_tf_keras_layer
@
ш0
щ1
ъ2
ы3"
trackable_list_wrapper
@
ш0
щ1
ъ2
ы3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
^	variables
_trainable_variables
`regularization_losses
b__call__
*c&call_and_return_all_conditional_losses
&c"call_and_return_conditional_losses"
_generic_user_object
┴
Еtrace_0
Жtrace_12Ж
%__inference_BPE_layer_call_fn_2997992
%__inference_BPE_layer_call_fn_2998005╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЕtrace_0zЖtrace_1
ў
Зtrace_0
Иtrace_12╝
@__inference_BPE_layer_call_and_return_conditional_losses_2997963
@__inference_BPE_layer_call_and_return_conditional_losses_2997979╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zЗtrace_0zИtrace_1
├
Й	variables
Кtrainable_variables
Лregularization_losses
М	keras_api
Н__call__
+О&call_and_return_all_conditional_losses
ьkernel
	эbias"
_tf_keras_layer
л
П	variables
Рtrainable_variables
Сregularization_losses
Т	keras_api
У__call__
+Ф&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Х	variables
Цtrainable_variables
Чregularization_losses
Ш	keras_api
Щ__call__
+Ъ&call_and_return_all_conditional_losses
юkernel
	яbias"
_tf_keras_layer
л
Ы	variables
Ьtrainable_variables
Эregularization_losses
Ю	keras_api
Я__call__
+а&call_and_return_all_conditional_losses"
_tf_keras_layer
@
ь0
э1
ю2
я3"
trackable_list_wrapper
@
ь0
э1
ю2
я3"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
бnon_trainable_variables
вlayers
гmetrics
 дlayer_regularization_losses
еlayer_metrics
h	variables
itrainable_variables
jregularization_losses
l__call__
*m&call_and_return_all_conditional_losses
&m"call_and_return_conditional_losses"
_generic_user_object
┴
жtrace_0
зtrace_12Ж
%__inference_BVE_layer_call_fn_2998148
%__inference_BVE_layer_call_fn_2998161╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zжtrace_0zзtrace_1
ў
иtrace_0
йtrace_12╝
@__inference_BVE_layer_call_and_return_conditional_losses_2998119
@__inference_BVE_layer_call_and_return_conditional_losses_2998135╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0zйtrace_1
`
Ё0
ё1
Є2
є3
Ї4
ї5
Ў6
ў7"
trackable_list_wrapper
`
Ё0
ё1
Є2
є3
Ї4
ї5
Ў6
ў7"
trackable_list_wrapper
 "
trackable_list_wrapper
▓
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
n	variables
otrainable_variables
pregularization_losses
r__call__
*s&call_and_return_all_conditional_losses
&s"call_and_return_conditional_losses"
_generic_user_object
╢
пtrace_0
░trace_12√
6__inference_multi_head_attention_layer_call_fn_2999752
6__inference_multi_head_attention_layer_call_fn_2999775И
Б▓¤
FullArgSpecp
argshЪe
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
defaultsв

 

 
p 
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zпtrace_0z░trace_1
ь
▒trace_0
▓trace_12▒
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2999811
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2999847И
Б▓¤
FullArgSpecp
argshЪe
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
defaultsв

 

 
p 
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z▒trace_0z▓trace_1
Ў
│	variables
┤trainable_variables
╡regularization_losses
╢	keras_api
╖__call__
+╕&call_and_return_all_conditional_losses
╣partial_output_shape
║full_output_shape
Ёkernel
	ёbias"
_tf_keras_layer
Ў
╗	variables
╝trainable_variables
╜regularization_losses
╛	keras_api
┐__call__
+└&call_and_return_all_conditional_losses
┴partial_output_shape
┬full_output_shape
Єkernel
	єbias"
_tf_keras_layer
Ў
├	variables
─trainable_variables
┼regularization_losses
╞	keras_api
╟__call__
+╚&call_and_return_all_conditional_losses
╔partial_output_shape
╩full_output_shape
Їkernel
	їbias"
_tf_keras_layer
л
╦	variables
╠trainable_variables
═regularization_losses
╬	keras_api
╧__call__
+╨&call_and_return_all_conditional_losses"
_tf_keras_layer
├
╤	variables
╥trainable_variables
╙regularization_losses
╘	keras_api
╒__call__
+╓&call_and_return_all_conditional_losses
╫_random_generator"
_tf_keras_layer
Ў
╪	variables
┘trainable_variables
┌regularization_losses
█	keras_api
▄__call__
+▌&call_and_return_all_conditional_losses
▐partial_output_shape
▀full_output_shape
Ўkernel
	ўbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
▓
рnon_trainable_variables
сlayers
тmetrics
 уlayer_regularization_losses
фlayer_metrics
z	variables
{trainable_variables
|regularization_losses
~__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
с
хtrace_02┬
%__inference_add_layer_call_fn_2999853Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zхtrace_0
№
цtrace_02▌
@__inference_add_layer_call_and_return_conditional_losses_2999859Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zцtrace_0
0
Ж0
З1"
trackable_list_wrapper
0
Ж0
З1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
чnon_trainable_variables
шlayers
щmetrics
 ъlayer_regularization_losses
ыlayer_metrics
А	variables
Бtrainable_variables
Вregularization_losses
Д__call__
+Е&call_and_return_all_conditional_losses
'Е"call_and_return_conditional_losses"
_generic_user_object
у
ьtrace_02─
'__inference_dense_layer_call_fn_2999868Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zьtrace_0
■
эtrace_02▀
B__inference_dense_layer_call_and_return_conditional_losses_2999899Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zэtrace_0
:@@2dense/kernel
:@2
dense/bias
0
О0
П1"
trackable_list_wrapper
0
О0
П1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
юnon_trainable_variables
яlayers
Ёmetrics
 ёlayer_regularization_losses
Єlayer_metrics
И	variables
Йtrainable_variables
Кregularization_losses
М__call__
+Н&call_and_return_all_conditional_losses
'Н"call_and_return_conditional_losses"
_generic_user_object
х
єtrace_02╞
)__inference_dense_1_layer_call_fn_2999908Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zєtrace_0
А
Їtrace_02с
D__inference_dense_1_layer_call_and_return_conditional_losses_2999939Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЇtrace_0
 :@@2dense_1/kernel
:@2dense_1/bias
`
°0
∙1
·2
√3
№4
¤5
■6
 7"
trackable_list_wrapper
`
°0
∙1
·2
√3
№4
¤5
■6
 7"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
їnon_trainable_variables
Ўlayers
ўmetrics
 °layer_regularization_losses
∙layer_metrics
Р	variables
Сtrainable_variables
Тregularization_losses
Ф__call__
+Х&call_and_return_all_conditional_losses
'Х"call_and_return_conditional_losses"
_generic_user_object
║
·trace_0
√trace_12 
8__inference_multi_head_attention_1_layer_call_fn_2999962
8__inference_multi_head_attention_1_layer_call_fn_2999985И
Б▓¤
FullArgSpecp
argshЪe
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
defaultsв

 

 
p 
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z·trace_0z√trace_1
Ё
№trace_0
¤trace_12╡
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_3000021
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_3000057И
Б▓¤
FullArgSpecp
argshЪe
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
defaultsв

 

 
p 
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z№trace_0z¤trace_1
Ў
■	variables
 trainable_variables
Аregularization_losses
Б	keras_api
В__call__
+Г&call_and_return_all_conditional_losses
Дpartial_output_shape
Еfull_output_shape
°kernel
	∙bias"
_tf_keras_layer
Ў
Ж	variables
Зtrainable_variables
Иregularization_losses
Й	keras_api
К__call__
+Л&call_and_return_all_conditional_losses
Мpartial_output_shape
Нfull_output_shape
·kernel
	√bias"
_tf_keras_layer
Ў
О	variables
Пtrainable_variables
Рregularization_losses
С	keras_api
Т__call__
+У&call_and_return_all_conditional_losses
Фpartial_output_shape
Хfull_output_shape
№kernel
	¤bias"
_tf_keras_layer
л
Ц	variables
Чtrainable_variables
Шregularization_losses
Щ	keras_api
Ъ__call__
+Ы&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Ь	variables
Эtrainable_variables
Юregularization_losses
Я	keras_api
а__call__
+б&call_and_return_all_conditional_losses
в_random_generator"
_tf_keras_layer
Ў
г	variables
дtrainable_variables
еregularization_losses
ж	keras_api
з__call__
+и&call_and_return_all_conditional_losses
йpartial_output_shape
кfull_output_shape
■kernel
	 bias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
лnon_trainable_variables
мlayers
нmetrics
 оlayer_regularization_losses
пlayer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
у
░trace_02─
'__inference_add_1_layer_call_fn_3000063Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z░trace_0
■
▒trace_02▀
B__inference_add_1_layer_call_and_return_conditional_losses_3000069Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z▒trace_0
0
и0
й1"
trackable_list_wrapper
0
и0
й1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▓non_trainable_variables
│layers
┤metrics
 ╡layer_regularization_losses
╢layer_metrics
в	variables
гtrainable_variables
дregularization_losses
ж__call__
+з&call_and_return_all_conditional_losses
'з"call_and_return_conditional_losses"
_generic_user_object
х
╖trace_02╞
)__inference_dense_2_layer_call_fn_3000078Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╖trace_0
А
╕trace_02с
D__inference_dense_2_layer_call_and_return_conditional_losses_3000109Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╕trace_0
 :@@2dense_2/kernel
:@2dense_2/bias
0
░0
▒1"
trackable_list_wrapper
0
░0
▒1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
к	variables
лtrainable_variables
мregularization_losses
о__call__
+п&call_and_return_all_conditional_losses
'п"call_and_return_conditional_losses"
_generic_user_object
х
╛trace_02╞
)__inference_dense_3_layer_call_fn_3000118Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╛trace_0
А
┐trace_02с
D__inference_dense_3_layer_call_and_return_conditional_losses_3000149Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z┐trace_0
 :@@2dense_3/kernel
:@2dense_3/bias
`
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7"
trackable_list_wrapper
`
А0
Б1
В2
Г3
Д4
Е5
Ж6
З7"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
└non_trainable_variables
┴layers
┬metrics
 ├layer_regularization_losses
─layer_metrics
▓	variables
│trainable_variables
┤regularization_losses
╢__call__
+╖&call_and_return_all_conditional_losses
'╖"call_and_return_conditional_losses"
_generic_user_object
║
┼trace_0
╞trace_12 
8__inference_multi_head_attention_2_layer_call_fn_3000172
8__inference_multi_head_attention_2_layer_call_fn_3000195И
Б▓¤
FullArgSpecp
argshЪe
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
defaultsв

 

 
p 
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z┼trace_0z╞trace_1
Ё
╟trace_0
╚trace_12╡
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_3000231
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_3000267И
Б▓¤
FullArgSpecp
argshЪe
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
defaultsв

 

 
p 
p 
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 z╟trace_0z╚trace_1
Ў
╔	variables
╩trainable_variables
╦regularization_losses
╠	keras_api
═__call__
+╬&call_and_return_all_conditional_losses
╧partial_output_shape
╨full_output_shape
Аkernel
	Бbias"
_tf_keras_layer
Ў
╤	variables
╥trainable_variables
╙regularization_losses
╘	keras_api
╒__call__
+╓&call_and_return_all_conditional_losses
╫partial_output_shape
╪full_output_shape
Вkernel
	Гbias"
_tf_keras_layer
Ў
┘	variables
┌trainable_variables
█regularization_losses
▄	keras_api
▌__call__
+▐&call_and_return_all_conditional_losses
▀partial_output_shape
рfull_output_shape
Дkernel
	Еbias"
_tf_keras_layer
л
с	variables
тtrainable_variables
уregularization_losses
ф	keras_api
х__call__
+ц&call_and_return_all_conditional_losses"
_tf_keras_layer
├
ч	variables
шtrainable_variables
щregularization_losses
ъ	keras_api
ы__call__
+ь&call_and_return_all_conditional_losses
э_random_generator"
_tf_keras_layer
Ў
ю	variables
яtrainable_variables
Ёregularization_losses
ё	keras_api
Є__call__
+є&call_and_return_all_conditional_losses
Їpartial_output_shape
їfull_output_shape
Жkernel
	Зbias"
_tf_keras_layer
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ўnon_trainable_variables
ўlayers
°metrics
 ∙layer_regularization_losses
·layer_metrics
╛	variables
┐trainable_variables
└regularization_losses
┬__call__
+├&call_and_return_all_conditional_losses
'├"call_and_return_conditional_losses"
_generic_user_object
у
√trace_02─
'__inference_add_2_layer_call_fn_3000273Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z√trace_0
■
№trace_02▀
B__inference_add_2_layer_call_and_return_conditional_losses_3000279Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z№trace_0
0
╩0
╦1"
trackable_list_wrapper
0
╩0
╦1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
¤non_trainable_variables
■layers
 metrics
 Аlayer_regularization_losses
Бlayer_metrics
─	variables
┼trainable_variables
╞regularization_losses
╚__call__
+╔&call_and_return_all_conditional_losses
'╔"call_and_return_conditional_losses"
_generic_user_object
х
Вtrace_02╞
)__inference_dense_4_layer_call_fn_3000288Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zВtrace_0
А
Гtrace_02с
D__inference_dense_4_layer_call_and_return_conditional_losses_3000319Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zГtrace_0
 :@@2dense_4/kernel
:@2dense_4/bias
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Дnon_trainable_variables
Еlayers
Жmetrics
 Зlayer_regularization_losses
Иlayer_metrics
╠	variables
═trainable_variables
╬regularization_losses
╨__call__
+╤&call_and_return_all_conditional_losses
'╤"call_and_return_conditional_losses"
_generic_user_object
х
Йtrace_02╞
)__inference_flatten_layer_call_fn_3000324Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЙtrace_0
А
Кtrace_02с
D__inference_flatten_layer_call_and_return_conditional_losses_3000330Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zКtrace_0
├
Л	variables
Мtrainable_variables
Нregularization_losses
О	keras_api
П__call__
+Р&call_and_return_all_conditional_losses
Иkernel
	Йbias"
_tf_keras_layer
л
С	variables
Тtrainable_variables
Уregularization_losses
Ф	keras_api
Х__call__
+Ц&call_and_return_all_conditional_losses"
_tf_keras_layer
├
Ч	variables
Шtrainable_variables
Щregularization_losses
Ъ	keras_api
Ы__call__
+Ь&call_and_return_all_conditional_losses
Кkernel
	Лbias"
_tf_keras_layer
л
Э	variables
Юtrainable_variables
Яregularization_losses
а	keras_api
б__call__
+в&call_and_return_all_conditional_losses"
_tf_keras_layer
@
И0
Й1
К2
Л3"
trackable_list_wrapper
@
И0
Й1
К2
Л3"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
гnon_trainable_variables
дlayers
еmetrics
 жlayer_regularization_losses
зlayer_metrics
╓	variables
╫trainable_variables
╪regularization_losses
┌__call__
+█&call_and_return_all_conditional_losses
'█"call_and_return_conditional_losses"
_generic_user_object
╜
иtrace_0
йtrace_12В
#__inference_U_layer_call_fn_2998264
#__inference_U_layer_call_fn_2998277╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zиtrace_0zйtrace_1
є
кtrace_0
лtrace_12╕
>__inference_U_layer_call_and_return_conditional_losses_2998235
>__inference_U_layer_call_and_return_conditional_losses_2998251╡
о▓к
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaultsв
p 

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 zкtrace_0zлtrace_1
0
т0
у1"
trackable_list_wrapper
0
т0
у1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
мnon_trainable_variables
нlayers
оmetrics
 пlayer_regularization_losses
░layer_metrics
▄	variables
▌trainable_variables
▐regularization_losses
р__call__
+с&call_and_return_all_conditional_losses
'с"call_and_return_conditional_losses"
_generic_user_object
ц
▒trace_02╟
*__inference_output_u_layer_call_fn_3000339Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z▒trace_0
Б
▓trace_02т
E__inference_output_u_layer_call_and_return_conditional_losses_3000349Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z▓trace_0
!:@2output_u/kernel
:2output_u/bias
':%@2spatial_layer1/kernel
!:@2spatial_layer1/bias
':%@@2spatial_layer2/kernel
!:@2spatial_layer2/bias
#:!@2bpe_layer1/kernel
:@2bpe_layer1/bias
#:!@@2bpe_layer2/kernel
:@2bpe_layer2/bias
#:!@2bve_layer1/kernel
:@2bve_layer1/bias
#:!@@2bve_layer2/kernel
:@2bve_layer2/bias
7:5@@2!multi_head_attention/query/kernel
1:/@2multi_head_attention/query/bias
5:3@@2multi_head_attention/key/kernel
/:-@2multi_head_attention/key/bias
7:5@@2!multi_head_attention/value/kernel
1:/@2multi_head_attention/value/bias
B:@@@2,multi_head_attention/attention_output/kernel
8:6@2*multi_head_attention/attention_output/bias
9:7@@2#multi_head_attention_1/query/kernel
3:1@2!multi_head_attention_1/query/bias
7:5@@2!multi_head_attention_1/key/kernel
1:/@2multi_head_attention_1/key/bias
9:7@@2#multi_head_attention_1/value/kernel
3:1@2!multi_head_attention_1/value/bias
D:B@@2.multi_head_attention_1/attention_output/kernel
::8@2,multi_head_attention_1/attention_output/bias
9:7@@2#multi_head_attention_2/query/kernel
3:1@2!multi_head_attention_2/query/bias
7:5@@2!multi_head_attention_2/key/kernel
1:/@2multi_head_attention_2/key/bias
9:7@@2#multi_head_attention_2/value/kernel
3:1@2!multi_head_attention_2/value/bias
D:B@@2.multi_head_attention_2/attention_output/kernel
::8@2,multi_head_attention_2/attention_output/bias
:@@2
ou1/kernel
:@2ou1/bias
:@@2
ou2/kernel
:@2ou2/bias
 "
trackable_list_wrapper
■
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
28"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
РBН
'__inference_model_layer_call_fn_2999154x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
РBН
'__inference_model_layer_call_fn_2999267x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
лBи
B__inference_model_layer_call_and_return_conditional_losses_2998797x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
лBи
B__inference_model_layer_call_and_return_conditional_losses_2999041x_inputt_input	Xbc_layer	tbc_layer	ubc_layer"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
лBи
%__inference_signature_wrapper_2999633	Xbc_layert_input	tbc_layer	ubc_layerx_input"╦
─▓└
FullArgSpec
argsЪ 
varargs
 
varkw
 
defaults
 N

kwonlyargs@Ъ=
j	Xbc_layer
	jt_input
j	tbc_layer
j	ubc_layer
	jx_input
kwonlydefaults
 
annotationsк *
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
╒B╥
+__inference_rescaling_layer_call_fn_2999638inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ЁBэ
F__inference_rescaling_layer_call_and_return_conditional_losses_2999646inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╫B╘
-__inference_rescaling_1_layer_call_fn_2999651inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ЄBя
H__inference_rescaling_1_layer_call_and_return_conditional_losses_2999659inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
уBр
-__inference_concatenate_layer_call_fn_2999665inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
■B√
H__inference_concatenate_layer_call_and_return_conditional_losses_2999672inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╫B╘
-__inference_rescaling_2_layer_call_fn_2999677inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ЄBя
H__inference_rescaling_2_layer_call_and_return_conditional_losses_2999685inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╫B╘
-__inference_rescaling_3_layer_call_fn_2999690inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ЄBя
H__inference_rescaling_3_layer_call_and_return_conditional_losses_2999698inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╙B╨
)__inference_reshape_layer_call_fn_2999703inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
юBы
D__inference_reshape_layer_call_and_return_conditional_losses_2999716inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
хBт
/__inference_concatenate_1_layer_call_fn_2999722inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
АB¤
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2999729inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
0
ф0
х1"
trackable_list_wrapper
0
ф0
х1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
│non_trainable_variables
┤layers
╡metrics
 ╢layer_regularization_losses
╖layer_metrics
╟	variables
╚trainable_variables
╔regularization_losses
╦__call__
+╠&call_and_return_all_conditional_losses
'╠"call_and_return_conditional_losses"
_generic_user_object
ь
╕trace_02═
0__inference_spatial_layer1_layer_call_fn_3000358Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╕trace_0
З
╣trace_02ш
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_3000388Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╣trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
║non_trainable_variables
╗layers
╝metrics
 ╜layer_regularization_losses
╛layer_metrics
═	variables
╬trainable_variables
╧regularization_losses
╤__call__
+╥&call_and_return_all_conditional_losses
'╥"call_and_return_conditional_losses"
_generic_user_object
ш
┐trace_02╔
,__inference_activation_layer_call_fn_3000393Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z┐trace_0
Г
└trace_02ф
G__inference_activation_layer_call_and_return_conditional_losses_3000398Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z└trace_0
0
ц0
ч1"
trackable_list_wrapper
0
ц0
ч1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
┴non_trainable_variables
┬layers
├metrics
 ─layer_regularization_losses
┼layer_metrics
╙	variables
╘trainable_variables
╒regularization_losses
╫__call__
+╪&call_and_return_all_conditional_losses
'╪"call_and_return_conditional_losses"
_generic_user_object
ь
╞trace_02═
0__inference_spatial_layer2_layer_call_fn_3000407Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╞trace_0
З
╟trace_02ш
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_3000437Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╟trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
┘	variables
┌trainable_variables
█regularization_losses
▌__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses"
_generic_user_object
ъ
═trace_02╦
.__inference_activation_1_layer_call_fn_3000442Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z═trace_0
Е
╬trace_02ц
I__inference_activation_1_layer_call_and_return_conditional_losses_3000447Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╬trace_0
 "
trackable_list_wrapper
<
P0
Q1
R2
S3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ДBБ
8__inference_spatial_transformation_layer_call_fn_2997836spatial_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ДBБ
8__inference_spatial_transformation_layer_call_fn_2997849spatial_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997807spatial_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ЯBЬ
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997823spatial_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
ш0
щ1"
trackable_list_wrapper
0
ш0
щ1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╧non_trainable_variables
╨layers
╤metrics
 ╥layer_regularization_losses
╙layer_metrics
ш	variables
щtrainable_variables
ъregularization_losses
ь__call__
+э&call_and_return_all_conditional_losses
'э"call_and_return_conditional_losses"
_generic_user_object
ш
╘trace_02╔
,__inference_bpe_layer1_layer_call_fn_3000456Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╘trace_0
Г
╒trace_02ф
G__inference_bpe_layer1_layer_call_and_return_conditional_losses_3000486Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z╒trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╓non_trainable_variables
╫layers
╪metrics
 ┘layer_regularization_losses
┌layer_metrics
ю	variables
яtrainable_variables
Ёregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
ъ
█trace_02╦
.__inference_activation_2_layer_call_fn_3000491Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z█trace_0
Е
▄trace_02ц
I__inference_activation_2_layer_call_and_return_conditional_losses_3000496Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z▄trace_0
0
ъ0
ы1"
trackable_list_wrapper
0
ъ0
ы1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▌non_trainable_variables
▐layers
▀metrics
 рlayer_regularization_losses
сlayer_metrics
Ї	variables
їtrainable_variables
Ўregularization_losses
°__call__
+∙&call_and_return_all_conditional_losses
'∙"call_and_return_conditional_losses"
_generic_user_object
ш
тtrace_02╔
,__inference_bpe_layer2_layer_call_fn_3000505Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zтtrace_0
Г
уtrace_02ф
G__inference_bpe_layer2_layer_call_and_return_conditional_losses_3000535Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zуtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
фnon_trainable_variables
хlayers
цmetrics
 чlayer_regularization_losses
шlayer_metrics
·	variables
√trainable_variables
№regularization_losses
■__call__
+ &call_and_return_all_conditional_losses
' "call_and_return_conditional_losses"
_generic_user_object
ъ
щtrace_02╦
.__inference_activation_3_layer_call_fn_3000540Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zщtrace_0
Е
ъtrace_02ц
I__inference_activation_3_layer_call_and_return_conditional_losses_3000545Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zъtrace_0
 "
trackable_list_wrapper
<
Z0
[1
\2
]3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
%__inference_BPE_layer_call_fn_2997992bpe_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
%__inference_BPE_layer_call_fn_2998005bpe_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
@__inference_BPE_layer_call_and_return_conditional_losses_2997963bpe_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
@__inference_BPE_layer_call_and_return_conditional_losses_2997979bpe_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
ь0
э1"
trackable_list_wrapper
0
ь0
э1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
ыnon_trainable_variables
ьlayers
эmetrics
 юlayer_regularization_losses
яlayer_metrics
Й	variables
Кtrainable_variables
Лregularization_losses
Н__call__
+О&call_and_return_all_conditional_losses
'О"call_and_return_conditional_losses"
_generic_user_object
ш
Ёtrace_02╔
,__inference_bve_layer1_layer_call_fn_3000554Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЁtrace_0
Г
ёtrace_02ф
G__inference_bve_layer1_layer_call_and_return_conditional_losses_3000584Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zёtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Єnon_trainable_variables
єlayers
Їmetrics
 їlayer_regularization_losses
Ўlayer_metrics
П	variables
Рtrainable_variables
Сregularization_losses
У__call__
+Ф&call_and_return_all_conditional_losses
'Ф"call_and_return_conditional_losses"
_generic_user_object
ъ
ўtrace_02╦
.__inference_activation_4_layer_call_fn_3000589Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zўtrace_0
Е
°trace_02ц
I__inference_activation_4_layer_call_and_return_conditional_losses_3000594Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z°trace_0
0
ю0
я1"
trackable_list_wrapper
0
ю0
я1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
∙non_trainable_variables
·layers
√metrics
 №layer_regularization_losses
¤layer_metrics
Х	variables
Цtrainable_variables
Чregularization_losses
Щ__call__
+Ъ&call_and_return_all_conditional_losses
'Ъ"call_and_return_conditional_losses"
_generic_user_object
ш
■trace_02╔
,__inference_bve_layer2_layer_call_fn_3000603Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z■trace_0
Г
 trace_02ф
G__inference_bve_layer2_layer_call_and_return_conditional_losses_3000633Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z trace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Аnon_trainable_variables
Бlayers
Вmetrics
 Гlayer_regularization_losses
Дlayer_metrics
Ы	variables
Ьtrainable_variables
Эregularization_losses
Я__call__
+а&call_and_return_all_conditional_losses
'а"call_and_return_conditional_losses"
_generic_user_object
ъ
Еtrace_02╦
.__inference_activation_5_layer_call_fn_3000638Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЕtrace_0
Е
Жtrace_02ц
I__inference_activation_5_layer_call_and_return_conditional_losses_3000643Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЖtrace_0
 "
trackable_list_wrapper
<
d0
e1
f2
g3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
эBъ
%__inference_BVE_layer_call_fn_2998148bve_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
эBъ
%__inference_BVE_layer_call_fn_2998161bve_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
@__inference_BVE_layer_call_and_return_conditional_losses_2998119bve_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
ИBЕ
@__inference_BVE_layer_call_and_return_conditional_losses_2998135bve_layer1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
J
t0
u1
v2
w3
x4
y5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╞B├
6__inference_multi_head_attention_layer_call_fn_2999752queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╞B├
6__inference_multi_head_attention_layer_call_fn_2999775queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
сB▐
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2999811queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
сB▐
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2999847queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
Ё0
ё1"
trackable_list_wrapper
0
Ё0
ё1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Зnon_trainable_variables
Иlayers
Йmetrics
 Кlayer_regularization_losses
Лlayer_metrics
│	variables
┤trainable_variables
╡regularization_losses
╖__call__
+╕&call_and_return_all_conditional_losses
'╕"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Є0
є1"
trackable_list_wrapper
0
Є0
є1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Мnon_trainable_variables
Нlayers
Оmetrics
 Пlayer_regularization_losses
Рlayer_metrics
╗	variables
╝trainable_variables
╜regularization_losses
┐__call__
+└&call_and_return_all_conditional_losses
'└"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Ї0
ї1"
trackable_list_wrapper
0
Ї0
ї1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Сnon_trainable_variables
Тlayers
Уmetrics
 Фlayer_regularization_losses
Хlayer_metrics
├	variables
─trainable_variables
┼regularization_losses
╟__call__
+╚&call_and_return_all_conditional_losses
'╚"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╕
Цnon_trainable_variables
Чlayers
Шmetrics
 Щlayer_regularization_losses
Ъlayer_metrics
╦	variables
╠trainable_variables
═regularization_losses
╧__call__
+╨&call_and_return_all_conditional_losses
'╨"call_and_return_conditional_losses"
_generic_user_object
л2ие
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
л2ие
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ыnon_trainable_variables
Ьlayers
Эmetrics
 Юlayer_regularization_losses
Яlayer_metrics
╤	variables
╥trainable_variables
╙regularization_losses
╒__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses"
_generic_user_object
п2мй
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2мй
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
0
Ў0
ў1"
trackable_list_wrapper
0
Ў0
ў1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
аnon_trainable_variables
бlayers
вmetrics
 гlayer_regularization_losses
дlayer_metrics
╪	variables
┘trainable_variables
┌regularization_losses
▄__call__
+▌&call_and_return_all_conditional_losses
'▌"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
█B╪
%__inference_add_layer_call_fn_2999853inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ЎBє
@__inference_add_layer_call_and_return_conditional_losses_2999859inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╤B╬
'__inference_dense_layer_call_fn_2999868inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ьBщ
B__inference_dense_layer_call_and_return_conditional_losses_2999899inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╙B╨
)__inference_dense_1_layer_call_fn_2999908inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
юBы
D__inference_dense_1_layer_call_and_return_conditional_losses_2999939inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
 "
trackable_list_wrapper
P
Ц0
Ч1
Ш2
Щ3
Ъ4
Ы5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╚B┼
8__inference_multi_head_attention_1_layer_call_fn_2999962queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╚B┼
8__inference_multi_head_attention_1_layer_call_fn_2999985queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
уBр
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_3000021queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
уBр
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_3000057queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
°0
∙1"
trackable_list_wrapper
0
°0
∙1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
еnon_trainable_variables
жlayers
зmetrics
 иlayer_regularization_losses
йlayer_metrics
■	variables
 trainable_variables
Аregularization_losses
В__call__
+Г&call_and_return_all_conditional_losses
'Г"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
·0
√1"
trackable_list_wrapper
0
·0
√1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
кnon_trainable_variables
лlayers
мmetrics
 нlayer_regularization_losses
оlayer_metrics
Ж	variables
Зtrainable_variables
Иregularization_losses
К__call__
+Л&call_and_return_all_conditional_losses
'Л"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
№0
¤1"
trackable_list_wrapper
0
№0
¤1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
пnon_trainable_variables
░layers
▒metrics
 ▓layer_regularization_losses
│layer_metrics
О	variables
Пtrainable_variables
Рregularization_losses
Т__call__
+У&call_and_return_all_conditional_losses
'У"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╕
┤non_trainable_variables
╡layers
╢metrics
 ╖layer_regularization_losses
╕layer_metrics
Ц	variables
Чtrainable_variables
Шregularization_losses
Ъ__call__
+Ы&call_and_return_all_conditional_losses
'Ы"call_and_return_conditional_losses"
_generic_user_object
л2ие
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
л2ие
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╣non_trainable_variables
║layers
╗metrics
 ╝layer_regularization_losses
╜layer_metrics
Ь	variables
Эtrainable_variables
Юregularization_losses
а__call__
+б&call_and_return_all_conditional_losses
'б"call_and_return_conditional_losses"
_generic_user_object
п2мй
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2мй
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
0
■0
 1"
trackable_list_wrapper
0
■0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╛non_trainable_variables
┐layers
└metrics
 ┴layer_regularization_losses
┬layer_metrics
г	variables
дtrainable_variables
еregularization_losses
з__call__
+и&call_and_return_all_conditional_losses
'и"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
▌B┌
'__inference_add_1_layer_call_fn_3000063inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
°Bї
B__inference_add_1_layer_call_and_return_conditional_losses_3000069inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╙B╨
)__inference_dense_2_layer_call_fn_3000078inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
юBы
D__inference_dense_2_layer_call_and_return_conditional_losses_3000109inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╙B╨
)__inference_dense_3_layer_call_fn_3000118inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
юBы
D__inference_dense_3_layer_call_and_return_conditional_losses_3000149inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
 "
trackable_list_wrapper
P
╕0
╣1
║2
╗3
╝4
╜5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
╚B┼
8__inference_multi_head_attention_2_layer_call_fn_3000172queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
╚B┼
8__inference_multi_head_attention_2_layer_call_fn_3000195queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
уBр
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_3000231queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
уBр
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_3000267queryvaluekey"є
ь▓ш
FullArgSpecp
argshЪe
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

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
0
А0
Б1"
trackable_list_wrapper
0
А0
Б1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
├non_trainable_variables
─layers
┼metrics
 ╞layer_regularization_losses
╟layer_metrics
╔	variables
╩trainable_variables
╦regularization_losses
═__call__
+╬&call_and_return_all_conditional_losses
'╬"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
0
В0
Г1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╚non_trainable_variables
╔layers
╩metrics
 ╦layer_regularization_losses
╠layer_metrics
╤	variables
╥trainable_variables
╙regularization_losses
╒__call__
+╓&call_and_return_all_conditional_losses
'╓"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
0
Д0
Е1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
═non_trainable_variables
╬layers
╧metrics
 ╨layer_regularization_losses
╤layer_metrics
┘	variables
┌trainable_variables
█regularization_losses
▌__call__
+▐&call_and_return_all_conditional_losses
'▐"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╕
╥non_trainable_variables
╙layers
╘metrics
 ╒layer_regularization_losses
╓layer_metrics
с	variables
тtrainable_variables
уregularization_losses
х__call__
+ц&call_and_return_all_conditional_losses
'ц"call_and_return_conditional_losses"
_generic_user_object
л2ие
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
л2ие
Ю▓Ъ
FullArgSpec
argsЪ
jinputs
jmask
varargs
 
varkw
 
defaultsв

 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
╫non_trainable_variables
╪layers
┘metrics
 ┌layer_regularization_losses
█layer_metrics
ч	variables
шtrainable_variables
щregularization_losses
ы__call__
+ь&call_and_return_all_conditional_losses
'ь"call_and_return_conditional_losses"
_generic_user_object
п2мй
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
п2мй
в▓Ю
FullArgSpec!
argsЪ
jinputs

jtraining
varargs
 
varkw
 
defaultsв
p 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
"
_generic_user_object
0
Ж0
З1"
trackable_list_wrapper
0
Ж0
З1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
▄non_trainable_variables
▌layers
▐metrics
 ▀layer_regularization_losses
рlayer_metrics
ю	variables
яtrainable_variables
Ёregularization_losses
Є__call__
+є&call_and_return_all_conditional_losses
'є"call_and_return_conditional_losses"
_generic_user_object
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
Ю2ЫШ
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
▌B┌
'__inference_add_2_layer_call_fn_3000273inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
°Bї
B__inference_add_2_layer_call_and_return_conditional_losses_3000279inputs_0inputs_1"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╙B╨
)__inference_dense_4_layer_call_fn_3000288inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
юBы
D__inference_dense_4_layer_call_and_return_conditional_losses_3000319inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╙B╨
)__inference_flatten_layer_call_fn_3000324inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
юBы
D__inference_flatten_layer_call_and_return_conditional_losses_3000330inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
0
И0
Й1"
trackable_list_wrapper
0
И0
Й1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
сnon_trainable_variables
тlayers
уmetrics
 фlayer_regularization_losses
хlayer_metrics
Л	variables
Мtrainable_variables
Нregularization_losses
П__call__
+Р&call_and_return_all_conditional_losses
'Р"call_and_return_conditional_losses"
_generic_user_object
с
цtrace_02┬
%__inference_ou1_layer_call_fn_3000652Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zцtrace_0
№
чtrace_02▌
@__inference_ou1_layer_call_and_return_conditional_losses_3000662Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zчtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
шnon_trainable_variables
щlayers
ъmetrics
 ыlayer_regularization_losses
ьlayer_metrics
С	variables
Тtrainable_variables
Уregularization_losses
Х__call__
+Ц&call_and_return_all_conditional_losses
'Ц"call_and_return_conditional_losses"
_generic_user_object
ъ
эtrace_02╦
.__inference_activation_6_layer_call_fn_3000667Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zэtrace_0
Е
юtrace_02ц
I__inference_activation_6_layer_call_and_return_conditional_losses_3000672Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zюtrace_0
0
К0
Л1"
trackable_list_wrapper
0
К0
Л1"
trackable_list_wrapper
 "
trackable_list_wrapper
╕
яnon_trainable_variables
Ёlayers
ёmetrics
 Єlayer_regularization_losses
єlayer_metrics
Ч	variables
Шtrainable_variables
Щregularization_losses
Ы__call__
+Ь&call_and_return_all_conditional_losses
'Ь"call_and_return_conditional_losses"
_generic_user_object
с
Їtrace_02┬
%__inference_ou2_layer_call_fn_3000681Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zЇtrace_0
№
їtrace_02▌
@__inference_ou2_layer_call_and_return_conditional_losses_3000691Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 zїtrace_0
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
╕
Ўnon_trainable_variables
ўlayers
°metrics
 ∙layer_regularization_losses
·layer_metrics
Э	variables
Юtrainable_variables
Яregularization_losses
б__call__
+в&call_and_return_all_conditional_losses
'в"call_and_return_conditional_losses"
_generic_user_object
ъ
√trace_02╦
.__inference_activation_7_layer_call_fn_3000696Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z√trace_0
Е
№trace_02ц
I__inference_activation_7_layer_call_and_return_conditional_losses_3000701Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 z№trace_0
 "
trackable_list_wrapper
@
╥0
╙1
╘2
╒3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
фBс
#__inference_U_layer_call_fn_2998264	ou1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
фBс
#__inference_U_layer_call_fn_2998277	ou1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
>__inference_U_layer_call_and_return_conditional_losses_2998235	ou1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
 
 B№
>__inference_U_layer_call_and_return_conditional_losses_2998251	ou1_input"м
е▓б
FullArgSpec)
args!Ъ
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
 

kwonlyargsЪ 
kwonlydefaults
 
annotationsк *
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
╘B╤
*__inference_output_u_layer_call_fn_3000339inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
яBь
E__inference_output_u_layer_call_and_return_conditional_losses_3000349inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
┌B╫
0__inference_spatial_layer1_layer_call_fn_3000358inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
їBЄ
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_3000388inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╓B╙
,__inference_activation_layer_call_fn_3000393inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ёBю
G__inference_activation_layer_call_and_return_conditional_losses_3000398inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
┌B╫
0__inference_spatial_layer2_layer_call_fn_3000407inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
їBЄ
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_3000437inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╪B╒
.__inference_activation_1_layer_call_fn_3000442inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
єBЁ
I__inference_activation_1_layer_call_and_return_conditional_losses_3000447inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╓B╙
,__inference_bpe_layer1_layer_call_fn_3000456inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ёBю
G__inference_bpe_layer1_layer_call_and_return_conditional_losses_3000486inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╪B╒
.__inference_activation_2_layer_call_fn_3000491inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
єBЁ
I__inference_activation_2_layer_call_and_return_conditional_losses_3000496inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╓B╙
,__inference_bpe_layer2_layer_call_fn_3000505inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ёBю
G__inference_bpe_layer2_layer_call_and_return_conditional_losses_3000535inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╪B╒
.__inference_activation_3_layer_call_fn_3000540inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
єBЁ
I__inference_activation_3_layer_call_and_return_conditional_losses_3000545inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╓B╙
,__inference_bve_layer1_layer_call_fn_3000554inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ёBю
G__inference_bve_layer1_layer_call_and_return_conditional_losses_3000584inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╪B╒
.__inference_activation_4_layer_call_fn_3000589inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
єBЁ
I__inference_activation_4_layer_call_and_return_conditional_losses_3000594inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╓B╙
,__inference_bve_layer2_layer_call_fn_3000603inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ёBю
G__inference_bve_layer2_layer_call_and_return_conditional_losses_3000633inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╪B╒
.__inference_activation_5_layer_call_fn_3000638inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
єBЁ
I__inference_activation_5_layer_call_and_return_conditional_losses_3000643inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╧B╠
%__inference_ou1_layer_call_fn_3000652inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ъBч
@__inference_ou1_layer_call_and_return_conditional_losses_3000662inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╪B╒
.__inference_activation_6_layer_call_fn_3000667inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
єBЁ
I__inference_activation_6_layer_call_and_return_conditional_losses_3000672inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╧B╠
%__inference_ou2_layer_call_fn_3000681inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
ъBч
@__inference_ou2_layer_call_and_return_conditional_losses_3000691inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
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
╪B╒
.__inference_activation_7_layer_call_fn_3000696inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 
єBЁ
I__inference_activation_7_layer_call_and_return_conditional_losses_3000701inputs"Ш
С▓Н
FullArgSpec
argsЪ

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
annotationsк *
 ┌
@__inference_BPE_layer_call_and_return_conditional_losses_2997963ХшщъыNвK
DвA
7К4
bpe_layer1_input                  
p

 
к "9в6
/К,
tensor_0                  @
Ъ ┌
@__inference_BPE_layer_call_and_return_conditional_losses_2997979ХшщъыNвK
DвA
7К4
bpe_layer1_input                  
p 

 
к "9в6
/К,
tensor_0                  @
Ъ ┤
%__inference_BPE_layer_call_fn_2997992КшщъыNвK
DвA
7К4
bpe_layer1_input                  
p

 
к ".К+
unknown                  @┤
%__inference_BPE_layer_call_fn_2998005КшщъыNвK
DвA
7К4
bpe_layer1_input                  
p 

 
к ".К+
unknown                  @┌
@__inference_BVE_layer_call_and_return_conditional_losses_2998119ХьэюяNвK
DвA
7К4
bve_layer1_input                  
p

 
к "9в6
/К,
tensor_0                  @
Ъ ┌
@__inference_BVE_layer_call_and_return_conditional_losses_2998135ХьэюяNвK
DвA
7К4
bve_layer1_input                  
p 

 
к "9в6
/К,
tensor_0                  @
Ъ ┤
%__inference_BVE_layer_call_fn_2998148КьэюяNвK
DвA
7К4
bve_layer1_input                  
p

 
к ".К+
unknown                  @┤
%__inference_BVE_layer_call_fn_2998161КьэюяNвK
DвA
7К4
bve_layer1_input                  
p 

 
к ".К+
unknown                  @╢
>__inference_U_layer_call_and_return_conditional_losses_2998235tИЙКЛ:в7
0в-
#К 
	ou1_input         @
p

 
к ",в)
"К
tensor_0         @
Ъ ╢
>__inference_U_layer_call_and_return_conditional_losses_2998251tИЙКЛ:в7
0в-
#К 
	ou1_input         @
p 

 
к ",в)
"К
tensor_0         @
Ъ Р
#__inference_U_layer_call_fn_2998264iИЙКЛ:в7
0в-
#К 
	ou1_input         @
p

 
к "!К
unknown         @Р
#__inference_U_layer_call_fn_2998277iИЙКЛ:в7
0в-
#К 
	ou1_input         @
p 

 
к "!К
unknown         @╝
"__inference__wrapped_model_2997721ХhшщъыьэюяфхцчЁёЄєЇїЎўЖЗОП°∙·√№¤■ ий░▒АБВГДЕЖЗ╩╦ИЙКЛтуєвя
чву
рЪ▄
!К
x_input         
!К
t_input         
0К-
	Xbc_layer                  
0К-
	tbc_layer                  
0К-
	ubc_layer                  
к "3к0
.
output_u"К
output_u         ┤
I__inference_activation_1_layer_call_and_return_conditional_losses_3000447g3в0
)в&
$К!
inputs         @
к "0в-
&К#
tensor_0         @
Ъ О
.__inference_activation_1_layer_call_fn_3000442\3в0
)в&
$К!
inputs         @
к "%К"
unknown         @╞
I__inference_activation_2_layer_call_and_return_conditional_losses_3000496y<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  @
Ъ а
.__inference_activation_2_layer_call_fn_3000491n<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  @╞
I__inference_activation_3_layer_call_and_return_conditional_losses_3000545y<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  @
Ъ а
.__inference_activation_3_layer_call_fn_3000540n<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  @╞
I__inference_activation_4_layer_call_and_return_conditional_losses_3000594y<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  @
Ъ а
.__inference_activation_4_layer_call_fn_3000589n<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  @╞
I__inference_activation_5_layer_call_and_return_conditional_losses_3000643y<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  @
Ъ а
.__inference_activation_5_layer_call_fn_3000638n<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  @м
I__inference_activation_6_layer_call_and_return_conditional_losses_3000672_/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         @
Ъ Ж
.__inference_activation_6_layer_call_fn_3000667T/в,
%в"
 К
inputs         @
к "!К
unknown         @м
I__inference_activation_7_layer_call_and_return_conditional_losses_3000701_/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         @
Ъ Ж
.__inference_activation_7_layer_call_fn_3000696T/в,
%в"
 К
inputs         @
к "!К
unknown         @▓
G__inference_activation_layer_call_and_return_conditional_losses_3000398g3в0
)в&
$К!
inputs         @
к "0в-
&К#
tensor_0         @
Ъ М
,__inference_activation_layer_call_fn_3000393\3в0
)в&
$К!
inputs         @
к "%К"
unknown         @▌
B__inference_add_1_layer_call_and_return_conditional_losses_3000069Цbв_
XвU
SЪP
&К#
inputs_0         @
&К#
inputs_1         @
к "0в-
&К#
tensor_0         @
Ъ ╖
'__inference_add_1_layer_call_fn_3000063Лbв_
XвU
SЪP
&К#
inputs_0         @
&К#
inputs_1         @
к "%К"
unknown         @▌
B__inference_add_2_layer_call_and_return_conditional_losses_3000279Цbв_
XвU
SЪP
&К#
inputs_0         @
&К#
inputs_1         @
к "0в-
&К#
tensor_0         @
Ъ ╖
'__inference_add_2_layer_call_fn_3000273Лbв_
XвU
SЪP
&К#
inputs_0         @
&К#
inputs_1         @
к "%К"
unknown         @█
@__inference_add_layer_call_and_return_conditional_losses_2999859Цbв_
XвU
SЪP
&К#
inputs_0         @
&К#
inputs_1         @
к "0в-
&К#
tensor_0         @
Ъ ╡
%__inference_add_layer_call_fn_2999853Лbв_
XвU
SЪP
&К#
inputs_0         @
&К#
inputs_1         @
к "%К"
unknown         @╩
G__inference_bpe_layer1_layer_call_and_return_conditional_losses_3000486шщ<в9
2в/
-К*
inputs                  
к "9в6
/К,
tensor_0                  @
Ъ д
,__inference_bpe_layer1_layer_call_fn_3000456tшщ<в9
2в/
-К*
inputs                  
к ".К+
unknown                  @╩
G__inference_bpe_layer2_layer_call_and_return_conditional_losses_3000535ъы<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  @
Ъ д
,__inference_bpe_layer2_layer_call_fn_3000505tъы<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  @╩
G__inference_bve_layer1_layer_call_and_return_conditional_losses_3000584ьэ<в9
2в/
-К*
inputs                  
к "9в6
/К,
tensor_0                  @
Ъ д
,__inference_bve_layer1_layer_call_fn_3000554tьэ<в9
2в/
-К*
inputs                  
к ".К+
unknown                  @╩
G__inference_bve_layer2_layer_call_and_return_conditional_losses_3000633юя<в9
2в/
-К*
inputs                  @
к "9в6
/К,
tensor_0                  @
Ъ д
,__inference_bve_layer2_layer_call_fn_3000603tюя<в9
2в/
-К*
inputs                  @
к ".К+
unknown                  @А
J__inference_concatenate_1_layer_call_and_return_conditional_losses_2999729▒tвq
jвg
eЪb
/К,
inputs_0                  
/К,
inputs_1                  
к "9в6
/К,
tensor_0                  
Ъ ┌
/__inference_concatenate_1_layer_call_fn_2999722жtвq
jвg
eЪb
/К,
inputs_0                  
/К,
inputs_1                  
к ".К+
unknown                  ╫
H__inference_concatenate_layer_call_and_return_conditional_losses_2999672КZвW
PвM
KЪH
"К
inputs_0         
"К
inputs_1         
к ",в)
"К
tensor_0         
Ъ ░
-__inference_concatenate_layer_call_fn_2999665ZвW
PвM
KЪH
"К
inputs_0         
"К
inputs_1         
к "!К
unknown         ╡
D__inference_dense_1_layer_call_and_return_conditional_losses_2999939mОП3в0
)в&
$К!
inputs         @
к "0в-
&К#
tensor_0         @
Ъ П
)__inference_dense_1_layer_call_fn_2999908bОП3в0
)в&
$К!
inputs         @
к "%К"
unknown         @╡
D__inference_dense_2_layer_call_and_return_conditional_losses_3000109mий3в0
)в&
$К!
inputs         @
к "0в-
&К#
tensor_0         @
Ъ П
)__inference_dense_2_layer_call_fn_3000078bий3в0
)в&
$К!
inputs         @
к "%К"
unknown         @╡
D__inference_dense_3_layer_call_and_return_conditional_losses_3000149m░▒3в0
)в&
$К!
inputs         @
к "0в-
&К#
tensor_0         @
Ъ П
)__inference_dense_3_layer_call_fn_3000118b░▒3в0
)в&
$К!
inputs         @
к "%К"
unknown         @╡
D__inference_dense_4_layer_call_and_return_conditional_losses_3000319m╩╦3в0
)в&
$К!
inputs         @
к "0в-
&К#
tensor_0         @
Ъ П
)__inference_dense_4_layer_call_fn_3000288b╩╦3в0
)в&
$К!
inputs         @
к "%К"
unknown         @│
B__inference_dense_layer_call_and_return_conditional_losses_2999899mЖЗ3в0
)в&
$К!
inputs         @
к "0в-
&К#
tensor_0         @
Ъ Н
'__inference_dense_layer_call_fn_2999868bЖЗ3в0
)в&
$К!
inputs         @
к "%К"
unknown         @л
D__inference_flatten_layer_call_and_return_conditional_losses_3000330c3в0
)в&
$К!
inputs         @
к ",в)
"К
tensor_0         @
Ъ Е
)__inference_flatten_layer_call_fn_3000324X3в0
)в&
$К!
inputs         @
к "!К
unknown         @▌
B__inference_model_layer_call_and_return_conditional_losses_2998797ЦhшщъыьэюяфхцчЁёЄєЇїЎўЖЗОП°∙·√№¤■ ий░▒АБВГДЕЖЗ╩╦ИЙКЛту√вў
явы
рЪ▄
!К
x_input         
!К
t_input         
0К-
	Xbc_layer                  
0К-
	tbc_layer                  
0К-
	ubc_layer                  
p

 
к ",в)
"К
tensor_0         
Ъ ▌
B__inference_model_layer_call_and_return_conditional_losses_2999041ЦhшщъыьэюяфхцчЁёЄєЇїЎўЖЗОП°∙·√№¤■ ий░▒АБВГДЕЖЗ╩╦ИЙКЛту√вў
явы
рЪ▄
!К
x_input         
!К
t_input         
0К-
	Xbc_layer                  
0К-
	tbc_layer                  
0К-
	ubc_layer                  
p 

 
к ",в)
"К
tensor_0         
Ъ ╖
'__inference_model_layer_call_fn_2999154ЛhшщъыьэюяфхцчЁёЄєЇїЎўЖЗОП°∙·√№¤■ ий░▒АБВГДЕЖЗ╩╦ИЙКЛту√вў
явы
рЪ▄
!К
x_input         
!К
t_input         
0К-
	Xbc_layer                  
0К-
	tbc_layer                  
0К-
	ubc_layer                  
p

 
к "!К
unknown         ╖
'__inference_model_layer_call_fn_2999267ЛhшщъыьэюяфхцчЁёЄєЇїЎўЖЗОП°∙·√№¤■ ий░▒АБВГДЕЖЗ╩╦ИЙКЛту√вў
явы
рЪ▄
!К
x_input         
!К
t_input         
0К-
	Xbc_layer                  
0К-
	tbc_layer                  
0К-
	ubc_layer                  
p 

 
к "!К
unknown         ╛
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_3000021ц°∙·√№¤■ ЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p
p 
к "0в-
&К#
tensor_0         @
Ъ ╛
S__inference_multi_head_attention_1_layer_call_and_return_conditional_losses_3000057ц°∙·√№¤■ ЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p 
p 
к "0в-
&К#
tensor_0         @
Ъ Ш
8__inference_multi_head_attention_1_layer_call_fn_2999962█°∙·√№¤■ ЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p
p 
к "%К"
unknown         @Ш
8__inference_multi_head_attention_1_layer_call_fn_2999985█°∙·√№¤■ ЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p 
p 
к "%К"
unknown         @╛
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_3000231цАБВГДЕЖЗЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p
p 
к "0в-
&К#
tensor_0         @
Ъ ╛
S__inference_multi_head_attention_2_layer_call_and_return_conditional_losses_3000267цАБВГДЕЖЗЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p 
p 
к "0в-
&К#
tensor_0         @
Ъ Ш
8__inference_multi_head_attention_2_layer_call_fn_3000172█АБВГДЕЖЗЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p
p 
к "%К"
unknown         @Ш
8__inference_multi_head_attention_2_layer_call_fn_3000195█АБВГДЕЖЗЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p 
p 
к "%К"
unknown         @╝
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2999811цЁёЄєЇїЎўЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p
p 
к "0в-
&К#
tensor_0         @
Ъ ╝
Q__inference_multi_head_attention_layer_call_and_return_conditional_losses_2999847цЁёЄєЇїЎўЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p 
p 
к "0в-
&К#
tensor_0         @
Ъ Ц
6__inference_multi_head_attention_layer_call_fn_2999752█ЁёЄєЇїЎўЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p
p 
к "%К"
unknown         @Ц
6__inference_multi_head_attention_layer_call_fn_2999775█ЁёЄєЇїЎўЯвЫ
УвП
#К 
query         @
,К)
value                  @
*К'
key                  @

 
p 
p 
p 
к "%К"
unknown         @й
@__inference_ou1_layer_call_and_return_conditional_losses_3000662eИЙ/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         @
Ъ Г
%__inference_ou1_layer_call_fn_3000652ZИЙ/в,
%в"
 К
inputs         @
к "!К
unknown         @й
@__inference_ou2_layer_call_and_return_conditional_losses_3000691eКЛ/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         @
Ъ Г
%__inference_ou2_layer_call_fn_3000681ZКЛ/в,
%в"
 К
inputs         @
к "!К
unknown         @о
E__inference_output_u_layer_call_and_return_conditional_losses_3000349eту/в,
%в"
 К
inputs         @
к ",в)
"К
tensor_0         
Ъ И
*__inference_output_u_layer_call_fn_3000339Zту/в,
%в"
 К
inputs         @
к "!К
unknown         л
H__inference_rescaling_1_layer_call_and_return_conditional_losses_2999659_/в,
%в"
 К
inputs         
к ",в)
"К
tensor_0         
Ъ Е
-__inference_rescaling_1_layer_call_fn_2999651T/в,
%в"
 К
inputs         
к "!К
unknown         ┼
H__inference_rescaling_2_layer_call_and_return_conditional_losses_2999685y<в9
2в/
-К*
inputs                  
к "9в6
/К,
tensor_0                  
Ъ Я
-__inference_rescaling_2_layer_call_fn_2999677n<в9
2в/
-К*
inputs                  
к ".К+
unknown                  ┼
H__inference_rescaling_3_layer_call_and_return_conditional_losses_2999698y<в9
2в/
-К*
inputs                  
к "9в6
/К,
tensor_0                  
Ъ Я
-__inference_rescaling_3_layer_call_fn_2999690n<в9
2в/
-К*
inputs                  
к ".К+
unknown                  й
F__inference_rescaling_layer_call_and_return_conditional_losses_2999646_/в,
%в"
 К
inputs         
к ",в)
"К
tensor_0         
Ъ Г
+__inference_rescaling_layer_call_fn_2999638T/в,
%в"
 К
inputs         
к "!К
unknown         л
D__inference_reshape_layer_call_and_return_conditional_losses_2999716c/в,
%в"
 К
inputs         
к "0в-
&К#
tensor_0         
Ъ Е
)__inference_reshape_layer_call_fn_2999703X/в,
%в"
 К
inputs         
к "%К"
unknown         ї
%__inference_signature_wrapper_2999633╦hшщъыьэюяфхцчЁёЄєЇїЎўЖЗОП°∙·√№¤■ ий░▒АБВГДЕЖЗ╩╦ИЙКЛтуйве
в 
ЭкЩ
=
	Xbc_layer0К-
	xbc_layer                  
,
t_input!К
t_input         
=
	tbc_layer0К-
	tbc_layer                  
=
	ubc_layer0К-
	ubc_layer                  
,
x_input!К
x_input         "3к0
.
output_u"К
output_u         ╝
K__inference_spatial_layer1_layer_call_and_return_conditional_losses_3000388mфх3в0
)в&
$К!
inputs         
к "0в-
&К#
tensor_0         @
Ъ Ц
0__inference_spatial_layer1_layer_call_fn_3000358bфх3в0
)в&
$К!
inputs         
к "%К"
unknown         @╝
K__inference_spatial_layer2_layer_call_and_return_conditional_losses_3000437mцч3в0
)в&
$К!
inputs         @
к "0в-
&К#
tensor_0         @
Ъ Ц
0__inference_spatial_layer2_layer_call_fn_3000407bцч3в0
)в&
$К!
inputs         @
к "%К"
unknown         @▀
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997807ЗфхцчIвF
?в<
2К/
spatial_layer1_input         
p

 
к "0в-
&К#
tensor_0         @
Ъ ▀
S__inference_spatial_transformation_layer_call_and_return_conditional_losses_2997823ЗфхцчIвF
?в<
2К/
spatial_layer1_input         
p 

 
к "0в-
&К#
tensor_0         @
Ъ ╕
8__inference_spatial_transformation_layer_call_fn_2997836|фхцчIвF
?в<
2К/
spatial_layer1_input         
p

 
к "%К"
unknown         @╕
8__inference_spatial_transformation_layer_call_fn_2997849|фхцчIвF
?в<
2К/
spatial_layer1_input         
p 

 
к "%К"
unknown         @