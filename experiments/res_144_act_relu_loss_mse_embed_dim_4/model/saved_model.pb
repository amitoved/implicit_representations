¬
Î£
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
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
¾
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
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.22v2.3.1-38-g9edbe5075f78Ù
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0

embedding_matrix/embeddingsVarHandleOp*
_output_shapes
: *
dtype0*
shape
:c*,
shared_nameembedding_matrix/embeddings

/embedding_matrix/embeddings/Read/ReadVariableOpReadVariableOpembedding_matrix/embeddings*
_output_shapes

:c*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0
z
dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_2/kernel
s
"dense_2/kernel/Read/ReadVariableOpReadVariableOpdense_2/kernel* 
_output_shapes
:
*
dtype0
q
dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_2/bias
j
 dense_2/bias/Read/ReadVariableOpReadVariableOpdense_2/bias*
_output_shapes	
:*
dtype0
z
dense_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_3/kernel
s
"dense_3/kernel/Read/ReadVariableOpReadVariableOpdense_3/kernel* 
_output_shapes
:
*
dtype0
q
dense_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_3/bias
j
 dense_3/bias/Read/ReadVariableOpReadVariableOpdense_3/bias*
_output_shapes	
:*
dtype0
z
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:*
dtype0
z
dense_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_5/kernel
s
"dense_5/kernel/Read/ReadVariableOpReadVariableOpdense_5/kernel* 
_output_shapes
:
*
dtype0
q
dense_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_5/bias
j
 dense_5/bias/Read/ReadVariableOpReadVariableOpdense_5/bias*
_output_shapes	
:*
dtype0
z
dense_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_6/kernel
s
"dense_6/kernel/Read/ReadVariableOpReadVariableOpdense_6/kernel* 
_output_shapes
:
*
dtype0
q
dense_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_6/bias
j
 dense_6/bias/Read/ReadVariableOpReadVariableOpdense_6/bias*
_output_shapes	
:*
dtype0
z
dense_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_7/kernel
s
"dense_7/kernel/Read/ReadVariableOpReadVariableOpdense_7/kernel* 
_output_shapes
:
*
dtype0
q
dense_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_7/bias
j
 dense_7/bias/Read/ReadVariableOpReadVariableOpdense_7/bias*
_output_shapes	
:*
dtype0
y
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*
shared_namedense_8/kernel
r
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel*
_output_shapes
:	*
dtype0
p
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_8/bias
i
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes
:*
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
 
"Adam/embedding_matrix/embeddings/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:c*3
shared_name$"Adam/embedding_matrix/embeddings/m

6Adam/embedding_matrix/embeddings/m/Read/ReadVariableOpReadVariableOp"Adam/embedding_matrix/embeddings/m*
_output_shapes

:c*
dtype0

Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	*
dtype0
{
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
t
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/m

)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/m
x
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_2/kernel/m

)Adam/dense_2/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/m
x
'Adam/dense_2/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_3/kernel/m

)Adam/dense_3/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_3/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/m
x
'Adam/dense_3/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_4/kernel/m

)Adam/dense_4/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_4/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/m
x
'Adam/dense_4/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_5/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_5/kernel/m

)Adam/dense_5/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_5/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/m
x
'Adam/dense_5/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_6/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_6/kernel/m

)Adam/dense_6/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_6/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/m
x
'Adam/dense_6/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_7/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_7/kernel/m

)Adam/dense_7/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/m* 
_output_shapes
:
*
dtype0

Adam/dense_7/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/m
x
'Adam/dense_7/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/m*
_output_shapes	
:*
dtype0

Adam/dense_8/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_8/kernel/m

)Adam/dense_8/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/m*
_output_shapes
:	*
dtype0
~
Adam/dense_8/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/m
w
'Adam/dense_8/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/m*
_output_shapes
:*
dtype0
 
"Adam/embedding_matrix/embeddings/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:c*3
shared_name$"Adam/embedding_matrix/embeddings/v

6Adam/embedding_matrix/embeddings/v/Read/ReadVariableOpReadVariableOp"Adam/embedding_matrix/embeddings/v*
_output_shapes

:c*
dtype0

Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	*
dtype0
{
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
t
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_1/kernel/v

)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_1/bias/v
x
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_2/kernel/v

)Adam/dense_2/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_2/bias/v
x
'Adam/dense_2/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_2/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_3/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_3/kernel/v

)Adam/dense_3/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_3/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_3/bias/v
x
'Adam/dense_3/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_3/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_4/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_4/kernel/v

)Adam/dense_4/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_4/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_4/bias/v
x
'Adam/dense_4/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_4/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_5/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_5/kernel/v

)Adam/dense_5/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_5/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_5/bias/v
x
'Adam/dense_5/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_5/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_6/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_6/kernel/v

)Adam/dense_6/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_6/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_6/bias/v
x
'Adam/dense_6/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_6/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_7/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*&
shared_nameAdam/dense_7/kernel/v

)Adam/dense_7/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/kernel/v* 
_output_shapes
:
*
dtype0

Adam/dense_7/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_7/bias/v
x
'Adam/dense_7/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_7/bias/v*
_output_shapes	
:*
dtype0

Adam/dense_8/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	*&
shared_nameAdam/dense_8/kernel/v

)Adam/dense_8/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/kernel/v*
_output_shapes
:	*
dtype0
~
Adam/dense_8/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*$
shared_nameAdam/dense_8/bias/v
w
'Adam/dense_8/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_8/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
o
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Õn
valueËnBÈn BÁn
þ
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
#_self_saveable_object_factories
	optimizer

signatures
trainable_variables
		variables

regularization_losses
	keras_api
%
#_self_saveable_object_factories
%
#_self_saveable_object_factories
¸
layer-0
layer_with_weights-0
layer-1
layer-2
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
ÿ
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
 layer_with_weights-7
 layer-10
!layer_with_weights-8
!layer-11
#"_self_saveable_object_factories
#trainable_variables
$	variables
%regularization_losses
&	keras_api
 
¼
'iter

(beta_1

)beta_2
	*decay
+learning_rate,mÎ-mÏ.mÐ/mÑ0mÒ1mÓ2mÔ3mÕ4mÖ5m×6mØ7mÙ8mÚ9mÛ:mÜ;mÝ<mÞ=mß>mà,vá-vâ.vã/vä0vå1væ2vç3vè4vé5vê6vë7vì8ví9vî:vï;vð<vñ=vò>vó
 

,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18

,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18
 
­

?layers
trainable_variables
@layer_regularization_losses
Alayer_metrics
Bnon_trainable_variables
Cmetrics
		variables

regularization_losses
 
 
%
#D_self_saveable_object_factories

,
embeddings
#E_self_saveable_object_factories
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
w
#J_self_saveable_object_factories
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
 

,0

,0
 
­

Olayers
trainable_variables
Player_regularization_losses
Qlayer_metrics
Rnon_trainable_variables
Smetrics
	variables
regularization_losses
%
#T_self_saveable_object_factories
%
#U_self_saveable_object_factories
w
#V_self_saveable_object_factories
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api


-kernel
.bias
#[_self_saveable_object_factories
\trainable_variables
]	variables
^regularization_losses
_	keras_api


/kernel
0bias
#`_self_saveable_object_factories
atrainable_variables
b	variables
cregularization_losses
d	keras_api


1kernel
2bias
#e_self_saveable_object_factories
ftrainable_variables
g	variables
hregularization_losses
i	keras_api


3kernel
4bias
#j_self_saveable_object_factories
ktrainable_variables
l	variables
mregularization_losses
n	keras_api


5kernel
6bias
#o_self_saveable_object_factories
ptrainable_variables
q	variables
rregularization_losses
s	keras_api


7kernel
8bias
#t_self_saveable_object_factories
utrainable_variables
v	variables
wregularization_losses
x	keras_api


9kernel
:bias
#y_self_saveable_object_factories
ztrainable_variables
{	variables
|regularization_losses
}	keras_api


;kernel
<bias
#~_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api


=kernel
>bias
$_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
 

-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17

-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17
 
²
layers
#trainable_variables
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
$	variables
%regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEembedding_matrix/embeddings0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense/kernel0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_1/kernel0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_1/bias0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_2/kernel0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_2/bias0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_3/kernel0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense_3/bias0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUEdense_4/kernel0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_4/bias1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_5/kernel1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_5/bias1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_6/kernel1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_6/bias1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_7/kernel1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_7/bias1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEdense_8/kernel1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEdense_8/bias1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
3
 
 
 

0
 
 

,0

,0
 
²
layers
 layer_regularization_losses
Ftrainable_variables
G	variables
non_trainable_variables
metrics
layer_metrics
Hregularization_losses
 
 
 
 
²
layers
 layer_regularization_losses
Ktrainable_variables
L	variables
non_trainable_variables
metrics
layer_metrics
Mregularization_losses

0
1
2
 
 
 
 
 
 
 
 
 
 
²
layers
 layer_regularization_losses
Wtrainable_variables
X	variables
non_trainable_variables
metrics
layer_metrics
Yregularization_losses
 

-0
.1

-0
.1
 
²
layers
 layer_regularization_losses
\trainable_variables
]	variables
non_trainable_variables
 metrics
¡layer_metrics
^regularization_losses
 

/0
01

/0
01
 
²
¢layers
 £layer_regularization_losses
atrainable_variables
b	variables
¤non_trainable_variables
¥metrics
¦layer_metrics
cregularization_losses
 

10
21

10
21
 
²
§layers
 ¨layer_regularization_losses
ftrainable_variables
g	variables
©non_trainable_variables
ªmetrics
«layer_metrics
hregularization_losses
 

30
41

30
41
 
²
¬layers
 ­layer_regularization_losses
ktrainable_variables
l	variables
®non_trainable_variables
¯metrics
°layer_metrics
mregularization_losses
 

50
61

50
61
 
²
±layers
 ²layer_regularization_losses
ptrainable_variables
q	variables
³non_trainable_variables
´metrics
µlayer_metrics
rregularization_losses
 

70
81

70
81
 
²
¶layers
 ·layer_regularization_losses
utrainable_variables
v	variables
¸non_trainable_variables
¹metrics
ºlayer_metrics
wregularization_losses
 

90
:1

90
:1
 
²
»layers
 ¼layer_regularization_losses
ztrainable_variables
{	variables
½non_trainable_variables
¾metrics
¿layer_metrics
|regularization_losses
 

;0
<1

;0
<1
 
´
Àlayers
 Álayer_regularization_losses
trainable_variables
	variables
Ânon_trainable_variables
Ãmetrics
Älayer_metrics
regularization_losses
 

=0
>1

=0
>1
 
µ
Ålayers
 Ælayer_regularization_losses
trainable_variables
	variables
Çnon_trainable_variables
Èmetrics
Élayer_metrics
regularization_losses
V
0
1
2
3
4
5
6
7
8
9
 10
!11
 
 
 
 
8

Êtotal

Ëcount
Ì	variables
Í	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

Ê0
Ë1

Ì	variables

VARIABLE_VALUE"Adam/embedding_matrix/embeddings/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_2/kernel/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_2/bias/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_3/kernel/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_3/bias/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_4/kernel/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_4/bias/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_5/kernel/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_5/bias/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_6/kernel/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_6/bias/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_7/kernel/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_7/bias/mMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_8/kernel/mMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_8/bias/mMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE

VARIABLE_VALUE"Adam/embedding_matrix/embeddings/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_1/kernel/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_1/bias/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_2/kernel/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_2/bias/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_3/kernel/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense_3/bias/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUEAdam/dense_4/kernel/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_4/bias/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_5/kernel/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_5/bias/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_6/kernel/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_6/bias/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_7/kernel/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_7/bias/vMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/dense_8/kernel/vMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/dense_8/bias/vMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
z
serving_default_input_1Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ
z
serving_default_input_2Placeholder*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype0*
shape:ÿÿÿÿÿÿÿÿÿ

StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1serving_default_input_2embedding_matrix/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *,
f'R%
#__inference_signature_wrapper_11540
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
¯
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp/embedding_matrix/embeddings/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp"dense_2/kernel/Read/ReadVariableOp dense_2/bias/Read/ReadVariableOp"dense_3/kernel/Read/ReadVariableOp dense_3/bias/Read/ReadVariableOp"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp"dense_5/kernel/Read/ReadVariableOp dense_5/bias/Read/ReadVariableOp"dense_6/kernel/Read/ReadVariableOp dense_6/bias/Read/ReadVariableOp"dense_7/kernel/Read/ReadVariableOp dense_7/bias/Read/ReadVariableOp"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp6Adam/embedding_matrix/embeddings/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp)Adam/dense_2/kernel/m/Read/ReadVariableOp'Adam/dense_2/bias/m/Read/ReadVariableOp)Adam/dense_3/kernel/m/Read/ReadVariableOp'Adam/dense_3/bias/m/Read/ReadVariableOp)Adam/dense_4/kernel/m/Read/ReadVariableOp'Adam/dense_4/bias/m/Read/ReadVariableOp)Adam/dense_5/kernel/m/Read/ReadVariableOp'Adam/dense_5/bias/m/Read/ReadVariableOp)Adam/dense_6/kernel/m/Read/ReadVariableOp'Adam/dense_6/bias/m/Read/ReadVariableOp)Adam/dense_7/kernel/m/Read/ReadVariableOp'Adam/dense_7/bias/m/Read/ReadVariableOp)Adam/dense_8/kernel/m/Read/ReadVariableOp'Adam/dense_8/bias/m/Read/ReadVariableOp6Adam/embedding_matrix/embeddings/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp)Adam/dense_2/kernel/v/Read/ReadVariableOp'Adam/dense_2/bias/v/Read/ReadVariableOp)Adam/dense_3/kernel/v/Read/ReadVariableOp'Adam/dense_3/bias/v/Read/ReadVariableOp)Adam/dense_4/kernel/v/Read/ReadVariableOp'Adam/dense_4/bias/v/Read/ReadVariableOp)Adam/dense_5/kernel/v/Read/ReadVariableOp'Adam/dense_5/bias/v/Read/ReadVariableOp)Adam/dense_6/kernel/v/Read/ReadVariableOp'Adam/dense_6/bias/v/Read/ReadVariableOp)Adam/dense_7/kernel/v/Read/ReadVariableOp'Adam/dense_7/bias/v/Read/ReadVariableOp)Adam/dense_8/kernel/v/Read/ReadVariableOp'Adam/dense_8/bias/v/Read/ReadVariableOpConst*M
TinF
D2B	*
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
GPU2*0J 8 *'
f"R 
__inference__traced_save_12561
ª
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rateembedding_matrix/embeddingsdense/kernel
dense/biasdense_1/kerneldense_1/biasdense_2/kerneldense_2/biasdense_3/kerneldense_3/biasdense_4/kerneldense_4/biasdense_5/kerneldense_5/biasdense_6/kerneldense_6/biasdense_7/kerneldense_7/biasdense_8/kerneldense_8/biastotalcount"Adam/embedding_matrix/embeddings/mAdam/dense/kernel/mAdam/dense/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/dense_2/kernel/mAdam/dense_2/bias/mAdam/dense_3/kernel/mAdam/dense_3/bias/mAdam/dense_4/kernel/mAdam/dense_4/bias/mAdam/dense_5/kernel/mAdam/dense_5/bias/mAdam/dense_6/kernel/mAdam/dense_6/bias/mAdam/dense_7/kernel/mAdam/dense_7/bias/mAdam/dense_8/kernel/mAdam/dense_8/bias/m"Adam/embedding_matrix/embeddings/vAdam/dense/kernel/vAdam/dense/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/dense_2/kernel/vAdam/dense_2/bias/vAdam/dense_3/kernel/vAdam/dense_3/bias/vAdam/dense_4/kernel/vAdam/dense_4/bias/vAdam/dense_5/kernel/vAdam/dense_5/bias/vAdam/dense_6/kernel/vAdam/dense_6/bias/vAdam/dense_7/kernel/vAdam/dense_7/bias/vAdam/dense_8/kernel/vAdam/dense_8/bias/v*L
TinE
C2A*
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
GPU2*0J 8 **
f%R#
!__inference__traced_restore_12763Ã


,__inference_functional_1_layer_call_fn_11480
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_114392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2

ì
G__inference_functional_1_layer_call_and_return_conditional_losses_11629
inputs_0
inputs_1;
7embedding_model_embedding_matrix_embedding_lookup_11545F
Bimplicit_representation_model_dense_matmul_readvariableop_resourceG
Cimplicit_representation_model_dense_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_1_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_1_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_2_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_2_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_3_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_3_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_4_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_4_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_5_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_5_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_6_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_6_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_7_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_7_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_8_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_8_biasadd_readvariableop_resource
identity¡
%embedding_model/embedding_matrix/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_model/embedding_matrix/Castò
1embedding_model/embedding_matrix/embedding_lookupResourceGather7embedding_model_embedding_matrix_embedding_lookup_11545)embedding_model/embedding_matrix/Cast:y:0*
Tindices0*J
_class@
><loc:@embedding_model/embedding_matrix/embedding_lookup/11545*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype023
1embedding_model/embedding_matrix/embedding_lookupÂ
:embedding_model/embedding_matrix/embedding_lookup/IdentityIdentity:embedding_model/embedding_matrix/embedding_lookup:output:0*
T0*J
_class@
><loc:@embedding_model/embedding_matrix/embedding_lookup/11545*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:embedding_model/embedding_matrix/embedding_lookup/Identity
<embedding_model/embedding_matrix/embedding_lookup/Identity_1IdentityCembedding_model/embedding_matrix/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<embedding_model/embedding_matrix/embedding_lookup/Identity_1³
embedding_model/reshape/ShapeShapeEembedding_model/embedding_matrix/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
embedding_model/reshape/Shape¤
+embedding_model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+embedding_model/reshape/strided_slice/stack¨
-embedding_model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-embedding_model/reshape/strided_slice/stack_1¨
-embedding_model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-embedding_model/reshape/strided_slice/stack_2ò
%embedding_model/reshape/strided_sliceStridedSlice&embedding_model/reshape/Shape:output:04embedding_model/reshape/strided_slice/stack:output:06embedding_model/reshape/strided_slice/stack_1:output:06embedding_model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%embedding_model/reshape/strided_slice
'embedding_model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'embedding_model/reshape/Reshape/shape/1æ
%embedding_model/reshape/Reshape/shapePack.embedding_model/reshape/strided_slice:output:00embedding_model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%embedding_model/reshape/Reshape/shapeö
embedding_model/reshape/ReshapeReshapeEembedding_model/embedding_matrix/embedding_lookup/Identity_1:output:0.embedding_model/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
embedding_model/reshape/Reshape°
5implicit_representation_model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :27
5implicit_representation_model/concatenate/concat/axis
0implicit_representation_model/concatenate/concatConcatV2inputs_0(embedding_model/reshape/Reshape:output:0>implicit_representation_model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0implicit_representation_model/concatenate/concatú
9implicit_representation_model/dense/MatMul/ReadVariableOpReadVariableOpBimplicit_representation_model_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02;
9implicit_representation_model/dense/MatMul/ReadVariableOp
*implicit_representation_model/dense/MatMulMatMul9implicit_representation_model/concatenate/concat:output:0Aimplicit_representation_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense/MatMulù
:implicit_representation_model/dense/BiasAdd/ReadVariableOpReadVariableOpCimplicit_representation_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:implicit_representation_model/dense/BiasAdd/ReadVariableOp
+implicit_representation_model/dense/BiasAddBiasAdd4implicit_representation_model/dense/MatMul:product:0Bimplicit_representation_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+implicit_representation_model/dense/BiasAddÅ
(implicit_representation_model/dense/ReluRelu4implicit_representation_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(implicit_representation_model/dense/Relu
;implicit_representation_model/dense_1/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_1/MatMul/ReadVariableOp
,implicit_representation_model/dense_1/MatMulMatMul6implicit_representation_model/dense/Relu:activations:0Cimplicit_representation_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_1/MatMulÿ
<implicit_representation_model/dense_1/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_1/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_1/BiasAddBiasAdd6implicit_representation_model/dense_1/MatMul:product:0Dimplicit_representation_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_1/BiasAddË
*implicit_representation_model/dense_1/ReluRelu6implicit_representation_model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_1/Relu
;implicit_representation_model/dense_2/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_2/MatMul/ReadVariableOp
,implicit_representation_model/dense_2/MatMulMatMul8implicit_representation_model/dense_1/Relu:activations:0Cimplicit_representation_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_2/MatMulÿ
<implicit_representation_model/dense_2/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_2/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_2/BiasAddBiasAdd6implicit_representation_model/dense_2/MatMul:product:0Dimplicit_representation_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_2/BiasAddË
*implicit_representation_model/dense_2/ReluRelu6implicit_representation_model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_2/Relu
;implicit_representation_model/dense_3/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_3/MatMul/ReadVariableOp
,implicit_representation_model/dense_3/MatMulMatMul8implicit_representation_model/dense_2/Relu:activations:0Cimplicit_representation_model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_3/MatMulÿ
<implicit_representation_model/dense_3/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_3/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_3/BiasAddBiasAdd6implicit_representation_model/dense_3/MatMul:product:0Dimplicit_representation_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_3/BiasAddË
*implicit_representation_model/dense_3/ReluRelu6implicit_representation_model/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_3/Relu
;implicit_representation_model/dense_4/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_4/MatMul/ReadVariableOp
,implicit_representation_model/dense_4/MatMulMatMul8implicit_representation_model/dense_3/Relu:activations:0Cimplicit_representation_model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_4/MatMulÿ
<implicit_representation_model/dense_4/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_4/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_4/BiasAddBiasAdd6implicit_representation_model/dense_4/MatMul:product:0Dimplicit_representation_model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_4/BiasAddË
*implicit_representation_model/dense_4/ReluRelu6implicit_representation_model/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_4/Relu
;implicit_representation_model/dense_5/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_5/MatMul/ReadVariableOp
,implicit_representation_model/dense_5/MatMulMatMul8implicit_representation_model/dense_4/Relu:activations:0Cimplicit_representation_model/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_5/MatMulÿ
<implicit_representation_model/dense_5/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_5/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_5/BiasAddBiasAdd6implicit_representation_model/dense_5/MatMul:product:0Dimplicit_representation_model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_5/BiasAddË
*implicit_representation_model/dense_5/ReluRelu6implicit_representation_model/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_5/Relu
;implicit_representation_model/dense_6/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_6/MatMul/ReadVariableOp
,implicit_representation_model/dense_6/MatMulMatMul8implicit_representation_model/dense_5/Relu:activations:0Cimplicit_representation_model/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_6/MatMulÿ
<implicit_representation_model/dense_6/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_6/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_6/BiasAddBiasAdd6implicit_representation_model/dense_6/MatMul:product:0Dimplicit_representation_model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_6/BiasAddË
*implicit_representation_model/dense_6/ReluRelu6implicit_representation_model/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_6/Relu
;implicit_representation_model/dense_7/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_7/MatMul/ReadVariableOp
,implicit_representation_model/dense_7/MatMulMatMul8implicit_representation_model/dense_6/Relu:activations:0Cimplicit_representation_model/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_7/MatMulÿ
<implicit_representation_model/dense_7/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_7/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_7/BiasAddBiasAdd6implicit_representation_model/dense_7/MatMul:product:0Dimplicit_representation_model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_7/BiasAddË
*implicit_representation_model/dense_7/ReluRelu6implicit_representation_model/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_7/Relu
;implicit_representation_model/dense_8/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02=
;implicit_representation_model/dense_8/MatMul/ReadVariableOp
,implicit_representation_model/dense_8/MatMulMatMul8implicit_representation_model/dense_7/Relu:activations:0Cimplicit_representation_model/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_8/MatMulþ
<implicit_representation_model/dense_8/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<implicit_representation_model/dense_8/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_8/BiasAddBiasAdd6implicit_representation_model/dense_8/MatMul:product:0Dimplicit_representation_model/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_8/BiasAddÔ
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOp7embedding_model_embedding_matrix_embedding_lookup_11545*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mul
IdentityIdentity6implicit_representation_model/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
°
ª
B__inference_dense_5_layer_call_and_return_conditional_losses_10753

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
|
'__inference_dense_2_layer_call_fn_12226

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_106722
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
|
'__inference_dense_1_layer_call_fn_12206

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_106452
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_6_layer_call_and_return_conditional_losses_10780

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

^
B__inference_reshape_layer_call_and_return_conditional_losses_10514

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


J__inference_embedding_model_layer_call_and_return_conditional_losses_11860

inputs+
'embedding_matrix_embedding_lookup_11840
identity
embedding_matrix/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_matrix/Cast¢
!embedding_matrix/embedding_lookupResourceGather'embedding_matrix_embedding_lookup_11840embedding_matrix/Cast:y:0*
Tindices0*:
_class0
.,loc:@embedding_matrix/embedding_lookup/11840*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02#
!embedding_matrix/embedding_lookup
*embedding_matrix/embedding_lookup/IdentityIdentity*embedding_matrix/embedding_lookup:output:0*
T0*:
_class0
.,loc:@embedding_matrix/embedding_lookup/11840*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*embedding_matrix/embedding_lookup/IdentityÓ
,embedding_matrix/embedding_lookup/Identity_1Identity3embedding_matrix/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,embedding_matrix/embedding_lookup/Identity_1
reshape/ShapeShape5embedding_matrix/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1¦
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape¶
reshape/ReshapeReshape5embedding_matrix/embedding_lookup/Identity_1:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/ReshapeÄ
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOp'embedding_matrix_embedding_lookup_11840*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mull
IdentityIdentityreshape/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ

#__inference_signature_wrapper_11540
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity¢StatefulPartitionedCallÃ
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *)
f$R"
 __inference__wrapped_model_104702
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2


K__inference_embedding_matrix_layer_call_and_return_conditional_losses_12118

inputs
embedding_lookup_12106
identity]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CastÍ
embedding_lookupResourceGatherembedding_lookup_12106Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/12106*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookup¾
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/12106*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1³
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_12106*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mul|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_3_layer_call_and_return_conditional_losses_10699

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
°
J__inference_embedding_model_layer_call_and_return_conditional_losses_10581

inputs
embedding_matrix_10570
identity¢(embedding_matrix/StatefulPartitionedCall¦
(embedding_matrix/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_matrix_10570*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_embedding_matrix_layer_call_and_return_conditional_losses_104902*
(embedding_matrix/StatefulPartitionedCallþ
reshape/PartitionedCallPartitionedCall1embedding_matrix/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_105142
reshape/PartitionedCall³
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_matrix_10570*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mul
IdentityIdentity reshape/PartitionedCall:output:0)^embedding_matrix/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2T
(embedding_matrix/StatefulPartitionedCall(embedding_matrix/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
¨
@__inference_dense_layer_call_and_return_conditional_losses_10618

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
¹
J__inference_embedding_model_layer_call_and_return_conditional_losses_10529
embedding_input
embedding_matrix_10499
identity¢(embedding_matrix/StatefulPartitionedCall¯
(embedding_matrix/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_matrix_10499*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_embedding_matrix_layer_call_and_return_conditional_losses_104902*
(embedding_matrix/StatefulPartitionedCallþ
reshape/PartitionedCallPartitionedCall1embedding_matrix/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_105142
reshape/PartitionedCall³
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_matrix_10499*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mul
IdentityIdentity reshape/PartitionedCall:output:0)^embedding_matrix/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2T
(embedding_matrix/StatefulPartitionedCall(embedding_matrix/StatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameembedding_input
°
ª
B__inference_dense_4_layer_call_and_return_conditional_losses_12257

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
|
'__inference_dense_3_layer_call_fn_12246

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_106992
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ë
u
/__inference_embedding_model_layer_call_fn_11867

inputs
unknown
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_model_layer_call_and_return_conditional_losses_105602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
|
'__inference_dense_5_layer_call_fn_12286

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_107532
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_4_layer_call_and_return_conditional_losses_10726

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
õ	
i
__inference_loss_fn_0_121539
5embeddings_regularizer_square_readvariableop_resource
identityÒ
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOp5embeddings_regularizer_square_readvariableop_resource*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mula
IdentityIdentityembeddings/Regularizer/mul:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:
Ù
z
%__inference_dense_layer_call_fn_12186

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallô
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_106182
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


,__inference_functional_1_layer_call_fn_11762
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_113442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1


,__inference_functional_1_layer_call_fn_11385
input_1
input_2
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity¢StatefulPartitionedCallê
StatefulPartitionedCallStatefulPartitionedCallinput_1input_2unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_113442
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
»
r
F__inference_concatenate_layer_call_and_return_conditional_losses_12160
inputs_0
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputs_0inputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1

!
!__inference__traced_restore_12763
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate2
.assignvariableop_5_embedding_matrix_embeddings#
assignvariableop_6_dense_kernel!
assignvariableop_7_dense_bias%
!assignvariableop_8_dense_1_kernel#
assignvariableop_9_dense_1_bias&
"assignvariableop_10_dense_2_kernel$
 assignvariableop_11_dense_2_bias&
"assignvariableop_12_dense_3_kernel$
 assignvariableop_13_dense_3_bias&
"assignvariableop_14_dense_4_kernel$
 assignvariableop_15_dense_4_bias&
"assignvariableop_16_dense_5_kernel$
 assignvariableop_17_dense_5_bias&
"assignvariableop_18_dense_6_kernel$
 assignvariableop_19_dense_6_bias&
"assignvariableop_20_dense_7_kernel$
 assignvariableop_21_dense_7_bias&
"assignvariableop_22_dense_8_kernel$
 assignvariableop_23_dense_8_bias
assignvariableop_24_total
assignvariableop_25_count:
6assignvariableop_26_adam_embedding_matrix_embeddings_m+
'assignvariableop_27_adam_dense_kernel_m)
%assignvariableop_28_adam_dense_bias_m-
)assignvariableop_29_adam_dense_1_kernel_m+
'assignvariableop_30_adam_dense_1_bias_m-
)assignvariableop_31_adam_dense_2_kernel_m+
'assignvariableop_32_adam_dense_2_bias_m-
)assignvariableop_33_adam_dense_3_kernel_m+
'assignvariableop_34_adam_dense_3_bias_m-
)assignvariableop_35_adam_dense_4_kernel_m+
'assignvariableop_36_adam_dense_4_bias_m-
)assignvariableop_37_adam_dense_5_kernel_m+
'assignvariableop_38_adam_dense_5_bias_m-
)assignvariableop_39_adam_dense_6_kernel_m+
'assignvariableop_40_adam_dense_6_bias_m-
)assignvariableop_41_adam_dense_7_kernel_m+
'assignvariableop_42_adam_dense_7_bias_m-
)assignvariableop_43_adam_dense_8_kernel_m+
'assignvariableop_44_adam_dense_8_bias_m:
6assignvariableop_45_adam_embedding_matrix_embeddings_v+
'assignvariableop_46_adam_dense_kernel_v)
%assignvariableop_47_adam_dense_bias_v-
)assignvariableop_48_adam_dense_1_kernel_v+
'assignvariableop_49_adam_dense_1_bias_v-
)assignvariableop_50_adam_dense_2_kernel_v+
'assignvariableop_51_adam_dense_2_bias_v-
)assignvariableop_52_adam_dense_3_kernel_v+
'assignvariableop_53_adam_dense_3_bias_v-
)assignvariableop_54_adam_dense_4_kernel_v+
'assignvariableop_55_adam_dense_4_bias_v-
)assignvariableop_56_adam_dense_5_kernel_v+
'assignvariableop_57_adam_dense_5_bias_v-
)assignvariableop_58_adam_dense_6_kernel_v+
'assignvariableop_59_adam_dense_6_bias_v-
)assignvariableop_60_adam_dense_7_kernel_v+
'assignvariableop_61_adam_dense_7_bias_v-
)assignvariableop_62_adam_dense_8_kernel_v+
'assignvariableop_63_adam_dense_8_bias_v
identity_65¢AssignVariableOp¢AssignVariableOp_1¢AssignVariableOp_10¢AssignVariableOp_11¢AssignVariableOp_12¢AssignVariableOp_13¢AssignVariableOp_14¢AssignVariableOp_15¢AssignVariableOp_16¢AssignVariableOp_17¢AssignVariableOp_18¢AssignVariableOp_19¢AssignVariableOp_2¢AssignVariableOp_20¢AssignVariableOp_21¢AssignVariableOp_22¢AssignVariableOp_23¢AssignVariableOp_24¢AssignVariableOp_25¢AssignVariableOp_26¢AssignVariableOp_27¢AssignVariableOp_28¢AssignVariableOp_29¢AssignVariableOp_3¢AssignVariableOp_30¢AssignVariableOp_31¢AssignVariableOp_32¢AssignVariableOp_33¢AssignVariableOp_34¢AssignVariableOp_35¢AssignVariableOp_36¢AssignVariableOp_37¢AssignVariableOp_38¢AssignVariableOp_39¢AssignVariableOp_4¢AssignVariableOp_40¢AssignVariableOp_41¢AssignVariableOp_42¢AssignVariableOp_43¢AssignVariableOp_44¢AssignVariableOp_45¢AssignVariableOp_46¢AssignVariableOp_47¢AssignVariableOp_48¢AssignVariableOp_49¢AssignVariableOp_5¢AssignVariableOp_50¢AssignVariableOp_51¢AssignVariableOp_52¢AssignVariableOp_53¢AssignVariableOp_54¢AssignVariableOp_55¢AssignVariableOp_56¢AssignVariableOp_57¢AssignVariableOp_58¢AssignVariableOp_59¢AssignVariableOp_6¢AssignVariableOp_60¢AssignVariableOp_61¢AssignVariableOp_62¢AssignVariableOp_63¢AssignVariableOp_7¢AssignVariableOp_8¢AssignVariableOp_9Ý"
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*é!
valueß!BÜ!AB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*
valueBAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slicesó
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*
_output_shapes
:::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*O
dtypesE
C2A	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1£
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2£
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3¢
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4ª
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5³
AssignVariableOp_5AssignVariableOp.assignvariableop_5_embedding_matrix_embeddingsIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6¤
AssignVariableOp_6AssignVariableOpassignvariableop_6_dense_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7¢
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8¦
AssignVariableOp_8AssignVariableOp!assignvariableop_8_dense_1_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9¤
AssignVariableOp_9AssignVariableOpassignvariableop_9_dense_1_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10ª
AssignVariableOp_10AssignVariableOp"assignvariableop_10_dense_2_kernelIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11¨
AssignVariableOp_11AssignVariableOp assignvariableop_11_dense_2_biasIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12ª
AssignVariableOp_12AssignVariableOp"assignvariableop_12_dense_3_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13¨
AssignVariableOp_13AssignVariableOp assignvariableop_13_dense_3_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14ª
AssignVariableOp_14AssignVariableOp"assignvariableop_14_dense_4_kernelIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15¨
AssignVariableOp_15AssignVariableOp assignvariableop_15_dense_4_biasIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16ª
AssignVariableOp_16AssignVariableOp"assignvariableop_16_dense_5_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17¨
AssignVariableOp_17AssignVariableOp assignvariableop_17_dense_5_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18ª
AssignVariableOp_18AssignVariableOp"assignvariableop_18_dense_6_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19¨
AssignVariableOp_19AssignVariableOp assignvariableop_19_dense_6_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20ª
AssignVariableOp_20AssignVariableOp"assignvariableop_20_dense_7_kernelIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21¨
AssignVariableOp_21AssignVariableOp assignvariableop_21_dense_7_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22ª
AssignVariableOp_22AssignVariableOp"assignvariableop_22_dense_8_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23¨
AssignVariableOp_23AssignVariableOp assignvariableop_23_dense_8_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24¡
AssignVariableOp_24AssignVariableOpassignvariableop_24_totalIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25¡
AssignVariableOp_25AssignVariableOpassignvariableop_25_countIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26¾
AssignVariableOp_26AssignVariableOp6assignvariableop_26_adam_embedding_matrix_embeddings_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27¯
AssignVariableOp_27AssignVariableOp'assignvariableop_27_adam_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28­
AssignVariableOp_28AssignVariableOp%assignvariableop_28_adam_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29±
AssignVariableOp_29AssignVariableOp)assignvariableop_29_adam_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30¯
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31±
AssignVariableOp_31AssignVariableOp)assignvariableop_31_adam_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32¯
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33±
AssignVariableOp_33AssignVariableOp)assignvariableop_33_adam_dense_3_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34¯
AssignVariableOp_34AssignVariableOp'assignvariableop_34_adam_dense_3_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35±
AssignVariableOp_35AssignVariableOp)assignvariableop_35_adam_dense_4_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36¯
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_4_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37±
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_5_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38¯
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_5_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39±
AssignVariableOp_39AssignVariableOp)assignvariableop_39_adam_dense_6_kernel_mIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40¯
AssignVariableOp_40AssignVariableOp'assignvariableop_40_adam_dense_6_bias_mIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41±
AssignVariableOp_41AssignVariableOp)assignvariableop_41_adam_dense_7_kernel_mIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42¯
AssignVariableOp_42AssignVariableOp'assignvariableop_42_adam_dense_7_bias_mIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43±
AssignVariableOp_43AssignVariableOp)assignvariableop_43_adam_dense_8_kernel_mIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44¯
AssignVariableOp_44AssignVariableOp'assignvariableop_44_adam_dense_8_bias_mIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45¾
AssignVariableOp_45AssignVariableOp6assignvariableop_45_adam_embedding_matrix_embeddings_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46¯
AssignVariableOp_46AssignVariableOp'assignvariableop_46_adam_dense_kernel_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47­
AssignVariableOp_47AssignVariableOp%assignvariableop_47_adam_dense_bias_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48±
AssignVariableOp_48AssignVariableOp)assignvariableop_48_adam_dense_1_kernel_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49¯
AssignVariableOp_49AssignVariableOp'assignvariableop_49_adam_dense_1_bias_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50±
AssignVariableOp_50AssignVariableOp)assignvariableop_50_adam_dense_2_kernel_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51¯
AssignVariableOp_51AssignVariableOp'assignvariableop_51_adam_dense_2_bias_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52±
AssignVariableOp_52AssignVariableOp)assignvariableop_52_adam_dense_3_kernel_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53¯
AssignVariableOp_53AssignVariableOp'assignvariableop_53_adam_dense_3_bias_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54±
AssignVariableOp_54AssignVariableOp)assignvariableop_54_adam_dense_4_kernel_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_54n
Identity_55IdentityRestoreV2:tensors:55"/device:CPU:0*
T0*
_output_shapes
:2
Identity_55¯
AssignVariableOp_55AssignVariableOp'assignvariableop_55_adam_dense_4_bias_vIdentity_55:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_55n
Identity_56IdentityRestoreV2:tensors:56"/device:CPU:0*
T0*
_output_shapes
:2
Identity_56±
AssignVariableOp_56AssignVariableOp)assignvariableop_56_adam_dense_5_kernel_vIdentity_56:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_56n
Identity_57IdentityRestoreV2:tensors:57"/device:CPU:0*
T0*
_output_shapes
:2
Identity_57¯
AssignVariableOp_57AssignVariableOp'assignvariableop_57_adam_dense_5_bias_vIdentity_57:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_57n
Identity_58IdentityRestoreV2:tensors:58"/device:CPU:0*
T0*
_output_shapes
:2
Identity_58±
AssignVariableOp_58AssignVariableOp)assignvariableop_58_adam_dense_6_kernel_vIdentity_58:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_58n
Identity_59IdentityRestoreV2:tensors:59"/device:CPU:0*
T0*
_output_shapes
:2
Identity_59¯
AssignVariableOp_59AssignVariableOp'assignvariableop_59_adam_dense_6_bias_vIdentity_59:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59n
Identity_60IdentityRestoreV2:tensors:60"/device:CPU:0*
T0*
_output_shapes
:2
Identity_60±
AssignVariableOp_60AssignVariableOp)assignvariableop_60_adam_dense_7_kernel_vIdentity_60:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_60n
Identity_61IdentityRestoreV2:tensors:61"/device:CPU:0*
T0*
_output_shapes
:2
Identity_61¯
AssignVariableOp_61AssignVariableOp'assignvariableop_61_adam_dense_7_bias_vIdentity_61:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_61n
Identity_62IdentityRestoreV2:tensors:62"/device:CPU:0*
T0*
_output_shapes
:2
Identity_62±
AssignVariableOp_62AssignVariableOp)assignvariableop_62_adam_dense_8_kernel_vIdentity_62:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_62n
Identity_63IdentityRestoreV2:tensors:63"/device:CPU:0*
T0*
_output_shapes
:2
Identity_63¯
AssignVariableOp_63AssignVariableOp'assignvariableop_63_adam_dense_8_bias_vIdentity_63:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_639
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpÞ
Identity_64Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_64Ñ
Identity_65IdentityIdentity_64:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_55^AssignVariableOp_56^AssignVariableOp_57^AssignVariableOp_58^AssignVariableOp_59^AssignVariableOp_6^AssignVariableOp_60^AssignVariableOp_61^AssignVariableOp_62^AssignVariableOp_63^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_65"#
identity_65Identity_65:output:0*
_input_shapes
: ::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
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
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
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
AssignVariableOp_4AssignVariableOp_42*
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
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542*
AssignVariableOp_55AssignVariableOp_552*
AssignVariableOp_56AssignVariableOp_562*
AssignVariableOp_57AssignVariableOp_572*
AssignVariableOp_58AssignVariableOp_582*
AssignVariableOp_59AssignVariableOp_592(
AssignVariableOp_6AssignVariableOp_62*
AssignVariableOp_60AssignVariableOp_602*
AssignVariableOp_61AssignVariableOp_612*
AssignVariableOp_62AssignVariableOp_622*
AssignVariableOp_63AssignVariableOp_632(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
Ù
º
 __inference__wrapped_model_10470
input_1
input_2H
Dfunctional_1_embedding_model_embedding_matrix_embedding_lookup_10392S
Ofunctional_1_implicit_representation_model_dense_matmul_readvariableop_resourceT
Pfunctional_1_implicit_representation_model_dense_biasadd_readvariableop_resourceU
Qfunctional_1_implicit_representation_model_dense_1_matmul_readvariableop_resourceV
Rfunctional_1_implicit_representation_model_dense_1_biasadd_readvariableop_resourceU
Qfunctional_1_implicit_representation_model_dense_2_matmul_readvariableop_resourceV
Rfunctional_1_implicit_representation_model_dense_2_biasadd_readvariableop_resourceU
Qfunctional_1_implicit_representation_model_dense_3_matmul_readvariableop_resourceV
Rfunctional_1_implicit_representation_model_dense_3_biasadd_readvariableop_resourceU
Qfunctional_1_implicit_representation_model_dense_4_matmul_readvariableop_resourceV
Rfunctional_1_implicit_representation_model_dense_4_biasadd_readvariableop_resourceU
Qfunctional_1_implicit_representation_model_dense_5_matmul_readvariableop_resourceV
Rfunctional_1_implicit_representation_model_dense_5_biasadd_readvariableop_resourceU
Qfunctional_1_implicit_representation_model_dense_6_matmul_readvariableop_resourceV
Rfunctional_1_implicit_representation_model_dense_6_biasadd_readvariableop_resourceU
Qfunctional_1_implicit_representation_model_dense_7_matmul_readvariableop_resourceV
Rfunctional_1_implicit_representation_model_dense_7_biasadd_readvariableop_resourceU
Qfunctional_1_implicit_representation_model_dense_8_matmul_readvariableop_resourceV
Rfunctional_1_implicit_representation_model_dense_8_biasadd_readvariableop_resource
identityº
2functional_1/embedding_model/embedding_matrix/CastCastinput_2*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ24
2functional_1/embedding_model/embedding_matrix/Cast³
>functional_1/embedding_model/embedding_matrix/embedding_lookupResourceGatherDfunctional_1_embedding_model_embedding_matrix_embedding_lookup_103926functional_1/embedding_model/embedding_matrix/Cast:y:0*
Tindices0*W
_classM
KIloc:@functional_1/embedding_model/embedding_matrix/embedding_lookup/10392*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02@
>functional_1/embedding_model/embedding_matrix/embedding_lookupö
Gfunctional_1/embedding_model/embedding_matrix/embedding_lookup/IdentityIdentityGfunctional_1/embedding_model/embedding_matrix/embedding_lookup:output:0*
T0*W
_classM
KIloc:@functional_1/embedding_model/embedding_matrix/embedding_lookup/10392*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2I
Gfunctional_1/embedding_model/embedding_matrix/embedding_lookup/Identityª
Ifunctional_1/embedding_model/embedding_matrix/embedding_lookup/Identity_1IdentityPfunctional_1/embedding_model/embedding_matrix/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2K
Ifunctional_1/embedding_model/embedding_matrix/embedding_lookup/Identity_1Ú
*functional_1/embedding_model/reshape/ShapeShapeRfunctional_1/embedding_model/embedding_matrix/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2,
*functional_1/embedding_model/reshape/Shape¾
8functional_1/embedding_model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2:
8functional_1/embedding_model/reshape/strided_slice/stackÂ
:functional_1/embedding_model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2<
:functional_1/embedding_model/reshape/strided_slice/stack_1Â
:functional_1/embedding_model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2<
:functional_1/embedding_model/reshape/strided_slice/stack_2À
2functional_1/embedding_model/reshape/strided_sliceStridedSlice3functional_1/embedding_model/reshape/Shape:output:0Afunctional_1/embedding_model/reshape/strided_slice/stack:output:0Cfunctional_1/embedding_model/reshape/strided_slice/stack_1:output:0Cfunctional_1/embedding_model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask24
2functional_1/embedding_model/reshape/strided_slice®
4functional_1/embedding_model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :26
4functional_1/embedding_model/reshape/Reshape/shape/1
2functional_1/embedding_model/reshape/Reshape/shapePack;functional_1/embedding_model/reshape/strided_slice:output:0=functional_1/embedding_model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:24
2functional_1/embedding_model/reshape/Reshape/shapeª
,functional_1/embedding_model/reshape/ReshapeReshapeRfunctional_1/embedding_model/embedding_matrix/embedding_lookup/Identity_1:output:0;functional_1/embedding_model/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,functional_1/embedding_model/reshape/ReshapeÊ
Bfunctional_1/implicit_representation_model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2D
Bfunctional_1/implicit_representation_model/concatenate/concat/axisÒ
=functional_1/implicit_representation_model/concatenate/concatConcatV2input_15functional_1/embedding_model/reshape/Reshape:output:0Kfunctional_1/implicit_representation_model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2?
=functional_1/implicit_representation_model/concatenate/concat¡
Ffunctional_1/implicit_representation_model/dense/MatMul/ReadVariableOpReadVariableOpOfunctional_1_implicit_representation_model_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02H
Ffunctional_1/implicit_representation_model/dense/MatMul/ReadVariableOpÇ
7functional_1/implicit_representation_model/dense/MatMulMatMulFfunctional_1/implicit_representation_model/concatenate/concat:output:0Nfunctional_1/implicit_representation_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7functional_1/implicit_representation_model/dense/MatMul 
Gfunctional_1/implicit_representation_model/dense/BiasAdd/ReadVariableOpReadVariableOpPfunctional_1_implicit_representation_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02I
Gfunctional_1/implicit_representation_model/dense/BiasAdd/ReadVariableOpÆ
8functional_1/implicit_representation_model/dense/BiasAddBiasAddAfunctional_1/implicit_representation_model/dense/MatMul:product:0Ofunctional_1/implicit_representation_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2:
8functional_1/implicit_representation_model/dense/BiasAddì
5functional_1/implicit_representation_model/dense/ReluReluAfunctional_1/implicit_representation_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ27
5functional_1/implicit_representation_model/dense/Relu¨
Hfunctional_1/implicit_representation_model/dense_1/MatMul/ReadVariableOpReadVariableOpQfunctional_1_implicit_representation_model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Hfunctional_1/implicit_representation_model/dense_1/MatMul/ReadVariableOpÊ
9functional_1/implicit_representation_model/dense_1/MatMulMatMulCfunctional_1/implicit_representation_model/dense/Relu:activations:0Pfunctional_1/implicit_representation_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9functional_1/implicit_representation_model/dense_1/MatMul¦
Ifunctional_1/implicit_representation_model/dense_1/BiasAdd/ReadVariableOpReadVariableOpRfunctional_1_implicit_representation_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifunctional_1/implicit_representation_model/dense_1/BiasAdd/ReadVariableOpÎ
:functional_1/implicit_representation_model/dense_1/BiasAddBiasAddCfunctional_1/implicit_representation_model/dense_1/MatMul:product:0Qfunctional_1/implicit_representation_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:functional_1/implicit_representation_model/dense_1/BiasAddò
7functional_1/implicit_representation_model/dense_1/ReluReluCfunctional_1/implicit_representation_model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7functional_1/implicit_representation_model/dense_1/Relu¨
Hfunctional_1/implicit_representation_model/dense_2/MatMul/ReadVariableOpReadVariableOpQfunctional_1_implicit_representation_model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Hfunctional_1/implicit_representation_model/dense_2/MatMul/ReadVariableOpÌ
9functional_1/implicit_representation_model/dense_2/MatMulMatMulEfunctional_1/implicit_representation_model/dense_1/Relu:activations:0Pfunctional_1/implicit_representation_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9functional_1/implicit_representation_model/dense_2/MatMul¦
Ifunctional_1/implicit_representation_model/dense_2/BiasAdd/ReadVariableOpReadVariableOpRfunctional_1_implicit_representation_model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifunctional_1/implicit_representation_model/dense_2/BiasAdd/ReadVariableOpÎ
:functional_1/implicit_representation_model/dense_2/BiasAddBiasAddCfunctional_1/implicit_representation_model/dense_2/MatMul:product:0Qfunctional_1/implicit_representation_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:functional_1/implicit_representation_model/dense_2/BiasAddò
7functional_1/implicit_representation_model/dense_2/ReluReluCfunctional_1/implicit_representation_model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7functional_1/implicit_representation_model/dense_2/Relu¨
Hfunctional_1/implicit_representation_model/dense_3/MatMul/ReadVariableOpReadVariableOpQfunctional_1_implicit_representation_model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Hfunctional_1/implicit_representation_model/dense_3/MatMul/ReadVariableOpÌ
9functional_1/implicit_representation_model/dense_3/MatMulMatMulEfunctional_1/implicit_representation_model/dense_2/Relu:activations:0Pfunctional_1/implicit_representation_model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9functional_1/implicit_representation_model/dense_3/MatMul¦
Ifunctional_1/implicit_representation_model/dense_3/BiasAdd/ReadVariableOpReadVariableOpRfunctional_1_implicit_representation_model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifunctional_1/implicit_representation_model/dense_3/BiasAdd/ReadVariableOpÎ
:functional_1/implicit_representation_model/dense_3/BiasAddBiasAddCfunctional_1/implicit_representation_model/dense_3/MatMul:product:0Qfunctional_1/implicit_representation_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:functional_1/implicit_representation_model/dense_3/BiasAddò
7functional_1/implicit_representation_model/dense_3/ReluReluCfunctional_1/implicit_representation_model/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7functional_1/implicit_representation_model/dense_3/Relu¨
Hfunctional_1/implicit_representation_model/dense_4/MatMul/ReadVariableOpReadVariableOpQfunctional_1_implicit_representation_model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Hfunctional_1/implicit_representation_model/dense_4/MatMul/ReadVariableOpÌ
9functional_1/implicit_representation_model/dense_4/MatMulMatMulEfunctional_1/implicit_representation_model/dense_3/Relu:activations:0Pfunctional_1/implicit_representation_model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9functional_1/implicit_representation_model/dense_4/MatMul¦
Ifunctional_1/implicit_representation_model/dense_4/BiasAdd/ReadVariableOpReadVariableOpRfunctional_1_implicit_representation_model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifunctional_1/implicit_representation_model/dense_4/BiasAdd/ReadVariableOpÎ
:functional_1/implicit_representation_model/dense_4/BiasAddBiasAddCfunctional_1/implicit_representation_model/dense_4/MatMul:product:0Qfunctional_1/implicit_representation_model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:functional_1/implicit_representation_model/dense_4/BiasAddò
7functional_1/implicit_representation_model/dense_4/ReluReluCfunctional_1/implicit_representation_model/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7functional_1/implicit_representation_model/dense_4/Relu¨
Hfunctional_1/implicit_representation_model/dense_5/MatMul/ReadVariableOpReadVariableOpQfunctional_1_implicit_representation_model_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Hfunctional_1/implicit_representation_model/dense_5/MatMul/ReadVariableOpÌ
9functional_1/implicit_representation_model/dense_5/MatMulMatMulEfunctional_1/implicit_representation_model/dense_4/Relu:activations:0Pfunctional_1/implicit_representation_model/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9functional_1/implicit_representation_model/dense_5/MatMul¦
Ifunctional_1/implicit_representation_model/dense_5/BiasAdd/ReadVariableOpReadVariableOpRfunctional_1_implicit_representation_model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifunctional_1/implicit_representation_model/dense_5/BiasAdd/ReadVariableOpÎ
:functional_1/implicit_representation_model/dense_5/BiasAddBiasAddCfunctional_1/implicit_representation_model/dense_5/MatMul:product:0Qfunctional_1/implicit_representation_model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:functional_1/implicit_representation_model/dense_5/BiasAddò
7functional_1/implicit_representation_model/dense_5/ReluReluCfunctional_1/implicit_representation_model/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7functional_1/implicit_representation_model/dense_5/Relu¨
Hfunctional_1/implicit_representation_model/dense_6/MatMul/ReadVariableOpReadVariableOpQfunctional_1_implicit_representation_model_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Hfunctional_1/implicit_representation_model/dense_6/MatMul/ReadVariableOpÌ
9functional_1/implicit_representation_model/dense_6/MatMulMatMulEfunctional_1/implicit_representation_model/dense_5/Relu:activations:0Pfunctional_1/implicit_representation_model/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9functional_1/implicit_representation_model/dense_6/MatMul¦
Ifunctional_1/implicit_representation_model/dense_6/BiasAdd/ReadVariableOpReadVariableOpRfunctional_1_implicit_representation_model_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifunctional_1/implicit_representation_model/dense_6/BiasAdd/ReadVariableOpÎ
:functional_1/implicit_representation_model/dense_6/BiasAddBiasAddCfunctional_1/implicit_representation_model/dense_6/MatMul:product:0Qfunctional_1/implicit_representation_model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:functional_1/implicit_representation_model/dense_6/BiasAddò
7functional_1/implicit_representation_model/dense_6/ReluReluCfunctional_1/implicit_representation_model/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7functional_1/implicit_representation_model/dense_6/Relu¨
Hfunctional_1/implicit_representation_model/dense_7/MatMul/ReadVariableOpReadVariableOpQfunctional_1_implicit_representation_model_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02J
Hfunctional_1/implicit_representation_model/dense_7/MatMul/ReadVariableOpÌ
9functional_1/implicit_representation_model/dense_7/MatMulMatMulEfunctional_1/implicit_representation_model/dense_6/Relu:activations:0Pfunctional_1/implicit_representation_model/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9functional_1/implicit_representation_model/dense_7/MatMul¦
Ifunctional_1/implicit_representation_model/dense_7/BiasAdd/ReadVariableOpReadVariableOpRfunctional_1_implicit_representation_model_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02K
Ifunctional_1/implicit_representation_model/dense_7/BiasAdd/ReadVariableOpÎ
:functional_1/implicit_representation_model/dense_7/BiasAddBiasAddCfunctional_1/implicit_representation_model/dense_7/MatMul:product:0Qfunctional_1/implicit_representation_model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:functional_1/implicit_representation_model/dense_7/BiasAddò
7functional_1/implicit_representation_model/dense_7/ReluReluCfunctional_1/implicit_representation_model/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ29
7functional_1/implicit_representation_model/dense_7/Relu§
Hfunctional_1/implicit_representation_model/dense_8/MatMul/ReadVariableOpReadVariableOpQfunctional_1_implicit_representation_model_dense_8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02J
Hfunctional_1/implicit_representation_model/dense_8/MatMul/ReadVariableOpË
9functional_1/implicit_representation_model/dense_8/MatMulMatMulEfunctional_1/implicit_representation_model/dense_7/Relu:activations:0Pfunctional_1/implicit_representation_model/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2;
9functional_1/implicit_representation_model/dense_8/MatMul¥
Ifunctional_1/implicit_representation_model/dense_8/BiasAdd/ReadVariableOpReadVariableOpRfunctional_1_implicit_representation_model_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02K
Ifunctional_1/implicit_representation_model/dense_8/BiasAdd/ReadVariableOpÍ
:functional_1/implicit_representation_model/dense_8/BiasAddBiasAddCfunctional_1/implicit_representation_model/dense_8/MatMul:product:0Qfunctional_1/implicit_representation_model/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:functional_1/implicit_representation_model/dense_8/BiasAdd
IdentityIdentityCfunctional_1/implicit_representation_model/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
°
ª
B__inference_dense_2_layer_call_and_return_conditional_losses_10672

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_7_layer_call_and_return_conditional_losses_10807

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
~
/__inference_embedding_model_layer_call_fn_10586
embedding_input
unknown
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_model_layer_call_and_return_conditional_losses_105812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameembedding_input


=__inference_implicit_representation_model_layer_call_fn_10995
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_109562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4
ØB
¤
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_11943
inputs_0
inputs_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identityt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis¥
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatenate/concat 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu§
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_3/MatMul/ReadVariableOp 
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¥
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¢
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAddq
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/Relu§
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/MatMul¥
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¢
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Relu§
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMul¥
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp¢
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/BiasAddq
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Relu§
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_6/MatMul/ReadVariableOp 
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¥
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¢
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Relu§
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_7/MatMul/ReadVariableOp 
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul¥
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¢
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Relu¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/BiasAddl
IdentityIdentitydense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
ï!
Ó
G__inference_functional_1_layer_call_and_return_conditional_losses_11439

inputs
inputs_1
embedding_model_11392'
#implicit_representation_model_11395'
#implicit_representation_model_11397'
#implicit_representation_model_11399'
#implicit_representation_model_11401'
#implicit_representation_model_11403'
#implicit_representation_model_11405'
#implicit_representation_model_11407'
#implicit_representation_model_11409'
#implicit_representation_model_11411'
#implicit_representation_model_11413'
#implicit_representation_model_11415'
#implicit_representation_model_11417'
#implicit_representation_model_11419'
#implicit_representation_model_11421'
#implicit_representation_model_11423'
#implicit_representation_model_11425'
#implicit_representation_model_11427'
#implicit_representation_model_11429
identity¢'embedding_model/StatefulPartitionedCall¢5implicit_representation_model/StatefulPartitionedCall 
'embedding_model/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_model_11392*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_model_layer_call_and_return_conditional_losses_105812)
'embedding_model/StatefulPartitionedCall 	
5implicit_representation_model/StatefulPartitionedCallStatefulPartitionedCallinputs0embedding_model/StatefulPartitionedCall:output:0#implicit_representation_model_11395#implicit_representation_model_11397#implicit_representation_model_11399#implicit_representation_model_11401#implicit_representation_model_11403#implicit_representation_model_11405#implicit_representation_model_11407#implicit_representation_model_11409#implicit_representation_model_11411#implicit_representation_model_11413#implicit_representation_model_11415#implicit_representation_model_11417#implicit_representation_model_11419#implicit_representation_model_11421#implicit_representation_model_11423#implicit_representation_model_11425#implicit_representation_model_11427#implicit_representation_model_11429*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_1104927
5implicit_representation_model/StatefulPartitionedCall²
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_model_11392*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mulô
IdentityIdentity>implicit_representation_model/StatefulPartitionedCall:output:0(^embedding_model/StatefulPartitionedCall6^implicit_representation_model/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::2R
'embedding_model/StatefulPartitionedCall'embedding_model/StatefulPartitionedCall2n
5implicit_representation_model/StatefulPartitionedCall5implicit_representation_model/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

ì
G__inference_functional_1_layer_call_and_return_conditional_losses_11718
inputs_0
inputs_1;
7embedding_model_embedding_matrix_embedding_lookup_11634F
Bimplicit_representation_model_dense_matmul_readvariableop_resourceG
Cimplicit_representation_model_dense_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_1_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_1_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_2_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_2_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_3_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_3_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_4_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_4_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_5_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_5_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_6_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_6_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_7_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_7_biasadd_readvariableop_resourceH
Dimplicit_representation_model_dense_8_matmul_readvariableop_resourceI
Eimplicit_representation_model_dense_8_biasadd_readvariableop_resource
identity¡
%embedding_model/embedding_matrix/CastCastinputs_1*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2'
%embedding_model/embedding_matrix/Castò
1embedding_model/embedding_matrix/embedding_lookupResourceGather7embedding_model_embedding_matrix_embedding_lookup_11634)embedding_model/embedding_matrix/Cast:y:0*
Tindices0*J
_class@
><loc:@embedding_model/embedding_matrix/embedding_lookup/11634*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype023
1embedding_model/embedding_matrix/embedding_lookupÂ
:embedding_model/embedding_matrix/embedding_lookup/IdentityIdentity:embedding_model/embedding_matrix/embedding_lookup:output:0*
T0*J
_class@
><loc:@embedding_model/embedding_matrix/embedding_lookup/11634*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2<
:embedding_model/embedding_matrix/embedding_lookup/Identity
<embedding_model/embedding_matrix/embedding_lookup/Identity_1IdentityCembedding_model/embedding_matrix/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2>
<embedding_model/embedding_matrix/embedding_lookup/Identity_1³
embedding_model/reshape/ShapeShapeEembedding_model/embedding_matrix/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
embedding_model/reshape/Shape¤
+embedding_model/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2-
+embedding_model/reshape/strided_slice/stack¨
-embedding_model/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2/
-embedding_model/reshape/strided_slice/stack_1¨
-embedding_model/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2/
-embedding_model/reshape/strided_slice/stack_2ò
%embedding_model/reshape/strided_sliceStridedSlice&embedding_model/reshape/Shape:output:04embedding_model/reshape/strided_slice/stack:output:06embedding_model/reshape/strided_slice/stack_1:output:06embedding_model/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2'
%embedding_model/reshape/strided_slice
'embedding_model/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2)
'embedding_model/reshape/Reshape/shape/1æ
%embedding_model/reshape/Reshape/shapePack.embedding_model/reshape/strided_slice:output:00embedding_model/reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2'
%embedding_model/reshape/Reshape/shapeö
embedding_model/reshape/ReshapeReshapeEembedding_model/embedding_matrix/embedding_lookup/Identity_1:output:0.embedding_model/reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2!
embedding_model/reshape/Reshape°
5implicit_representation_model/concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :27
5implicit_representation_model/concatenate/concat/axis
0implicit_representation_model/concatenate/concatConcatV2inputs_0(embedding_model/reshape/Reshape:output:0>implicit_representation_model/concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ22
0implicit_representation_model/concatenate/concatú
9implicit_representation_model/dense/MatMul/ReadVariableOpReadVariableOpBimplicit_representation_model_dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02;
9implicit_representation_model/dense/MatMul/ReadVariableOp
*implicit_representation_model/dense/MatMulMatMul9implicit_representation_model/concatenate/concat:output:0Aimplicit_representation_model/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense/MatMulù
:implicit_representation_model/dense/BiasAdd/ReadVariableOpReadVariableOpCimplicit_representation_model_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02<
:implicit_representation_model/dense/BiasAdd/ReadVariableOp
+implicit_representation_model/dense/BiasAddBiasAdd4implicit_representation_model/dense/MatMul:product:0Bimplicit_representation_model/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2-
+implicit_representation_model/dense/BiasAddÅ
(implicit_representation_model/dense/ReluRelu4implicit_representation_model/dense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2*
(implicit_representation_model/dense/Relu
;implicit_representation_model/dense_1/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_1/MatMul/ReadVariableOp
,implicit_representation_model/dense_1/MatMulMatMul6implicit_representation_model/dense/Relu:activations:0Cimplicit_representation_model/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_1/MatMulÿ
<implicit_representation_model/dense_1/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_1/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_1/BiasAddBiasAdd6implicit_representation_model/dense_1/MatMul:product:0Dimplicit_representation_model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_1/BiasAddË
*implicit_representation_model/dense_1/ReluRelu6implicit_representation_model/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_1/Relu
;implicit_representation_model/dense_2/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_2/MatMul/ReadVariableOp
,implicit_representation_model/dense_2/MatMulMatMul8implicit_representation_model/dense_1/Relu:activations:0Cimplicit_representation_model/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_2/MatMulÿ
<implicit_representation_model/dense_2/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_2/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_2/BiasAddBiasAdd6implicit_representation_model/dense_2/MatMul:product:0Dimplicit_representation_model/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_2/BiasAddË
*implicit_representation_model/dense_2/ReluRelu6implicit_representation_model/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_2/Relu
;implicit_representation_model/dense_3/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_3/MatMul/ReadVariableOp
,implicit_representation_model/dense_3/MatMulMatMul8implicit_representation_model/dense_2/Relu:activations:0Cimplicit_representation_model/dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_3/MatMulÿ
<implicit_representation_model/dense_3/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_3/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_3/BiasAddBiasAdd6implicit_representation_model/dense_3/MatMul:product:0Dimplicit_representation_model/dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_3/BiasAddË
*implicit_representation_model/dense_3/ReluRelu6implicit_representation_model/dense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_3/Relu
;implicit_representation_model/dense_4/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_4/MatMul/ReadVariableOp
,implicit_representation_model/dense_4/MatMulMatMul8implicit_representation_model/dense_3/Relu:activations:0Cimplicit_representation_model/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_4/MatMulÿ
<implicit_representation_model/dense_4/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_4/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_4/BiasAddBiasAdd6implicit_representation_model/dense_4/MatMul:product:0Dimplicit_representation_model/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_4/BiasAddË
*implicit_representation_model/dense_4/ReluRelu6implicit_representation_model/dense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_4/Relu
;implicit_representation_model/dense_5/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_5/MatMul/ReadVariableOp
,implicit_representation_model/dense_5/MatMulMatMul8implicit_representation_model/dense_4/Relu:activations:0Cimplicit_representation_model/dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_5/MatMulÿ
<implicit_representation_model/dense_5/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_5/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_5/BiasAddBiasAdd6implicit_representation_model/dense_5/MatMul:product:0Dimplicit_representation_model/dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_5/BiasAddË
*implicit_representation_model/dense_5/ReluRelu6implicit_representation_model/dense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_5/Relu
;implicit_representation_model/dense_6/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_6/MatMul/ReadVariableOp
,implicit_representation_model/dense_6/MatMulMatMul8implicit_representation_model/dense_5/Relu:activations:0Cimplicit_representation_model/dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_6/MatMulÿ
<implicit_representation_model/dense_6/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_6/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_6/BiasAddBiasAdd6implicit_representation_model/dense_6/MatMul:product:0Dimplicit_representation_model/dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_6/BiasAddË
*implicit_representation_model/dense_6/ReluRelu6implicit_representation_model/dense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_6/Relu
;implicit_representation_model/dense_7/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02=
;implicit_representation_model/dense_7/MatMul/ReadVariableOp
,implicit_representation_model/dense_7/MatMulMatMul8implicit_representation_model/dense_6/Relu:activations:0Cimplicit_representation_model/dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_7/MatMulÿ
<implicit_representation_model/dense_7/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02>
<implicit_representation_model/dense_7/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_7/BiasAddBiasAdd6implicit_representation_model/dense_7/MatMul:product:0Dimplicit_representation_model/dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_7/BiasAddË
*implicit_representation_model/dense_7/ReluRelu6implicit_representation_model/dense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*implicit_representation_model/dense_7/Relu
;implicit_representation_model/dense_8/MatMul/ReadVariableOpReadVariableOpDimplicit_representation_model_dense_8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02=
;implicit_representation_model/dense_8/MatMul/ReadVariableOp
,implicit_representation_model/dense_8/MatMulMatMul8implicit_representation_model/dense_7/Relu:activations:0Cimplicit_representation_model/dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,implicit_representation_model/dense_8/MatMulþ
<implicit_representation_model/dense_8/BiasAdd/ReadVariableOpReadVariableOpEimplicit_representation_model_dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02>
<implicit_representation_model/dense_8/BiasAdd/ReadVariableOp
-implicit_representation_model/dense_8/BiasAddBiasAdd6implicit_representation_model/dense_8/MatMul:product:0Dimplicit_representation_model/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2/
-implicit_representation_model/dense_8/BiasAddÔ
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOp7embedding_model_embedding_matrix_embedding_lookup_11634*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mul
IdentityIdentity6implicit_representation_model/dense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::::Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
â|
ý
__inference__traced_save_12561
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop:
6savev2_embedding_matrix_embeddings_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop-
)savev2_dense_2_kernel_read_readvariableop+
'savev2_dense_2_bias_read_readvariableop-
)savev2_dense_3_kernel_read_readvariableop+
'savev2_dense_3_bias_read_readvariableop-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop-
)savev2_dense_5_kernel_read_readvariableop+
'savev2_dense_5_bias_read_readvariableop-
)savev2_dense_6_kernel_read_readvariableop+
'savev2_dense_6_bias_read_readvariableop-
)savev2_dense_7_kernel_read_readvariableop+
'savev2_dense_7_bias_read_readvariableop-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopA
=savev2_adam_embedding_matrix_embeddings_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop4
0savev2_adam_dense_2_kernel_m_read_readvariableop2
.savev2_adam_dense_2_bias_m_read_readvariableop4
0savev2_adam_dense_3_kernel_m_read_readvariableop2
.savev2_adam_dense_3_bias_m_read_readvariableop4
0savev2_adam_dense_4_kernel_m_read_readvariableop2
.savev2_adam_dense_4_bias_m_read_readvariableop4
0savev2_adam_dense_5_kernel_m_read_readvariableop2
.savev2_adam_dense_5_bias_m_read_readvariableop4
0savev2_adam_dense_6_kernel_m_read_readvariableop2
.savev2_adam_dense_6_bias_m_read_readvariableop4
0savev2_adam_dense_7_kernel_m_read_readvariableop2
.savev2_adam_dense_7_bias_m_read_readvariableop4
0savev2_adam_dense_8_kernel_m_read_readvariableop2
.savev2_adam_dense_8_bias_m_read_readvariableopA
=savev2_adam_embedding_matrix_embeddings_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop4
0savev2_adam_dense_2_kernel_v_read_readvariableop2
.savev2_adam_dense_2_bias_v_read_readvariableop4
0savev2_adam_dense_3_kernel_v_read_readvariableop2
.savev2_adam_dense_3_bias_v_read_readvariableop4
0savev2_adam_dense_4_kernel_v_read_readvariableop2
.savev2_adam_dense_4_bias_v_read_readvariableop4
0savev2_adam_dense_5_kernel_v_read_readvariableop2
.savev2_adam_dense_5_bias_v_read_readvariableop4
0savev2_adam_dense_6_kernel_v_read_readvariableop2
.savev2_adam_dense_6_bias_v_read_readvariableop4
0savev2_adam_dense_7_kernel_v_read_readvariableop2
.savev2_adam_dense_7_bias_v_read_readvariableop4
0savev2_adam_dense_8_kernel_v_read_readvariableop2
.savev2_adam_dense_8_bias_v_read_readvariableop
savev2_const

identity_1¢MergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_3116ec8f3b544b0caa4c4ab805c5e203/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard¦
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename×"
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*é!
valueß!BÜ!AB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/16/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/17/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/18/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/16/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/17/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/18/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:A*
dtype0*
valueBAB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop6savev2_embedding_matrix_embeddings_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop)savev2_dense_2_kernel_read_readvariableop'savev2_dense_2_bias_read_readvariableop)savev2_dense_3_kernel_read_readvariableop'savev2_dense_3_bias_read_readvariableop)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop)savev2_dense_5_kernel_read_readvariableop'savev2_dense_5_bias_read_readvariableop)savev2_dense_6_kernel_read_readvariableop'savev2_dense_6_bias_read_readvariableop)savev2_dense_7_kernel_read_readvariableop'savev2_dense_7_bias_read_readvariableop)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop=savev2_adam_embedding_matrix_embeddings_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop0savev2_adam_dense_2_kernel_m_read_readvariableop.savev2_adam_dense_2_bias_m_read_readvariableop0savev2_adam_dense_3_kernel_m_read_readvariableop.savev2_adam_dense_3_bias_m_read_readvariableop0savev2_adam_dense_4_kernel_m_read_readvariableop.savev2_adam_dense_4_bias_m_read_readvariableop0savev2_adam_dense_5_kernel_m_read_readvariableop.savev2_adam_dense_5_bias_m_read_readvariableop0savev2_adam_dense_6_kernel_m_read_readvariableop.savev2_adam_dense_6_bias_m_read_readvariableop0savev2_adam_dense_7_kernel_m_read_readvariableop.savev2_adam_dense_7_bias_m_read_readvariableop0savev2_adam_dense_8_kernel_m_read_readvariableop.savev2_adam_dense_8_bias_m_read_readvariableop=savev2_adam_embedding_matrix_embeddings_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop0savev2_adam_dense_2_kernel_v_read_readvariableop.savev2_adam_dense_2_bias_v_read_readvariableop0savev2_adam_dense_3_kernel_v_read_readvariableop.savev2_adam_dense_3_bias_v_read_readvariableop0savev2_adam_dense_4_kernel_v_read_readvariableop.savev2_adam_dense_4_bias_v_read_readvariableop0savev2_adam_dense_5_kernel_v_read_readvariableop.savev2_adam_dense_5_bias_v_read_readvariableop0savev2_adam_dense_6_kernel_v_read_readvariableop.savev2_adam_dense_6_bias_v_read_readvariableop0savev2_adam_dense_7_kernel_v_read_readvariableop.savev2_adam_dense_7_bias_v_read_readvariableop0savev2_adam_dense_8_kernel_v_read_readvariableop.savev2_adam_dense_8_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *O
dtypesE
C2A	2
SaveV2º
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes¡
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*½
_input_shapes«
¨: : : : : : :c:	::
::
::
::
::
::
::
::	:: : :c:	::
::
::
::
::
::
::
::	::c:	::
::
::
::
::
::
::
::	:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:c:%!

_output_shapes
:	:!

_output_shapes	
::&	"
 
_output_shapes
:
:!


_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::%!

_output_shapes
:	: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:c:%!

_output_shapes
:	:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::& "
 
_output_shapes
:
:!!

_output_shapes	
::&""
 
_output_shapes
:
:!#

_output_shapes	
::&$"
 
_output_shapes
:
:!%

_output_shapes	
::&&"
 
_output_shapes
:
:!'

_output_shapes	
::&("
 
_output_shapes
:
:!)

_output_shapes	
::&*"
 
_output_shapes
:
:!+

_output_shapes	
::%,!

_output_shapes
:	: -

_output_shapes
::$. 

_output_shapes

:c:%/!

_output_shapes
:	:!0

_output_shapes	
::&1"
 
_output_shapes
:
:!2

_output_shapes	
::&3"
 
_output_shapes
:
:!4

_output_shapes	
::&5"
 
_output_shapes
:
:!6

_output_shapes	
::&7"
 
_output_shapes
:
:!8

_output_shapes	
::&9"
 
_output_shapes
:
:!:

_output_shapes	
::&;"
 
_output_shapes
:
:!<

_output_shapes	
::&="
 
_output_shapes
:
:!>

_output_shapes	
::%?!

_output_shapes
:	: @

_output_shapes
::A

_output_shapes
: 
°
ª
B__inference_dense_5_layer_call_and_return_conditional_losses_12277

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_1_layer_call_and_return_conditional_losses_12197

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
|
'__inference_dense_6_layer_call_fn_12306

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_107802
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


=__inference_implicit_representation_model_layer_call_fn_11088
input_3
input_4
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallí
StatefulPartitionedCallStatefulPartitionedCallinput_3input_4unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_110492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4


=__inference_implicit_representation_model_layer_call_fn_12096
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_110492
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
2

X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_10850
input_3
input_4
dense_10629
dense_10631
dense_1_10656
dense_1_10658
dense_2_10683
dense_2_10685
dense_3_10710
dense_3_10712
dense_4_10737
dense_4_10739
dense_5_10764
dense_5_10766
dense_6_10791
dense_6_10793
dense_7_10818
dense_7_10820
dense_8_10844
dense_8_10846
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCallê
concatenate/PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_105982
concatenate/PartitionedCall¤
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_10629dense_10631*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_106182
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_10656dense_1_10658*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_106452!
dense_1/StatefulPartitionedCall²
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_10683dense_2_10685*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_106722!
dense_2/StatefulPartitionedCall²
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_10710dense_3_10712*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_106992!
dense_3/StatefulPartitionedCall²
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_10737dense_4_10739*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_107262!
dense_4/StatefulPartitionedCall²
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_10764dense_5_10766*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_107532!
dense_5/StatefulPartitionedCall²
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_10791dense_6_10793*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_107802!
dense_6/StatefulPartitionedCall²
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_10818dense_7_10820*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_108072!
dense_7/StatefulPartitionedCall±
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_10844dense_8_10846*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_108332!
dense_8/StatefulPartitionedCall¬
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4


K__inference_embedding_matrix_layer_call_and_return_conditional_losses_10490

inputs
embedding_lookup_10478
identity]
CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
CastÍ
embedding_lookupResourceGatherembedding_lookup_10478Cast:y:0*
Tindices0*)
_class
loc:@embedding_lookup/10478*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02
embedding_lookup¾
embedding_lookup/IdentityIdentityembedding_lookup:output:0*
T0*)
_class
loc:@embedding_lookup/10478*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity 
embedding_lookup/Identity_1Identity"embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_lookup/Identity_1³
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_lookup_10478*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mul|
IdentityIdentity$embedding_lookup/Identity_1:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_3_layer_call_and_return_conditional_losses_12237

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
æ
~
/__inference_embedding_model_layer_call_fn_10565
embedding_input
unknown
identity¢StatefulPartitionedCallù
StatefulPartitionedCallStatefulPartitionedCallembedding_inputunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_model_layer_call_and_return_conditional_losses_105602
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameembedding_input
2

X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_11049

inputs
inputs_1
dense_11003
dense_11005
dense_1_11008
dense_1_11010
dense_2_11013
dense_2_11015
dense_3_11018
dense_3_11020
dense_4_11023
dense_4_11025
dense_5_11028
dense_5_11030
dense_6_11033
dense_6_11035
dense_7_11038
dense_7_11040
dense_8_11043
dense_8_11045
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCallê
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_105982
concatenate/PartitionedCall¤
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_11003dense_11005*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_106182
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_11008dense_1_11010*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_106452!
dense_1/StatefulPartitionedCall²
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_11013dense_2_11015*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_106722!
dense_2/StatefulPartitionedCall²
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_11018dense_3_11020*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_106992!
dense_3/StatefulPartitionedCall²
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_11023dense_4_11025*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_107262!
dense_4/StatefulPartitionedCall²
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_11028dense_5_11030*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_107532!
dense_5/StatefulPartitionedCall²
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_11033dense_6_11035*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_107802!
dense_6/StatefulPartitionedCall²
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_11038dense_7_11040*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_108072!
dense_7/StatefulPartitionedCall±
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_11043dense_8_11045*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_108332!
dense_8/StatefulPartitionedCall¬
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs


J__inference_embedding_model_layer_call_and_return_conditional_losses_11836

inputs+
'embedding_matrix_embedding_lookup_11816
identity
embedding_matrix/CastCastinputs*

DstT0*

SrcT0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
embedding_matrix/Cast¢
!embedding_matrix/embedding_lookupResourceGather'embedding_matrix_embedding_lookup_11816embedding_matrix/Cast:y:0*
Tindices0*:
_class0
.,loc:@embedding_matrix/embedding_lookup/11816*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*
dtype02#
!embedding_matrix/embedding_lookup
*embedding_matrix/embedding_lookup/IdentityIdentity*embedding_matrix/embedding_lookup:output:0*
T0*:
_class0
.,loc:@embedding_matrix/embedding_lookup/11816*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2,
*embedding_matrix/embedding_lookup/IdentityÓ
,embedding_matrix/embedding_lookup/Identity_1Identity3embedding_matrix/embedding_lookup/Identity:output:0*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2.
,embedding_matrix/embedding_lookup/Identity_1
reshape/ShapeShape5embedding_matrix/embedding_lookup/Identity_1:output:0*
T0*
_output_shapes
:2
reshape/Shape
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1¦
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape¶
reshape/ReshapeReshape5embedding_matrix/embedding_lookup/Identity_1:output:0reshape/Reshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
reshape/ReshapeÄ
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOp'embedding_matrix_embedding_lookup_11816*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mull
IdentityIdentityreshape/Reshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_2_layer_call_and_return_conditional_losses_12217

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ØB
¤
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_12012
inputs_0
inputs_1(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource*
&dense_3_matmul_readvariableop_resource+
'dense_3_biasadd_readvariableop_resource*
&dense_4_matmul_readvariableop_resource+
'dense_4_biasadd_readvariableop_resource*
&dense_5_matmul_readvariableop_resource+
'dense_5_biasadd_readvariableop_resource*
&dense_6_matmul_readvariableop_resource+
'dense_6_biasadd_readvariableop_resource*
&dense_7_matmul_readvariableop_resource+
'dense_7_biasadd_readvariableop_resource*
&dense_8_matmul_readvariableop_resource+
'dense_8_biasadd_readvariableop_resource
identityt
concatenate/concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concatenate/concat/axis¥
concatenate/concatConcatV2inputs_0inputs_1 concatenate/concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatenate/concat 
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulconcatenate/concat:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense/BiasAddk

dense/ReluReludense/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

dense/Relu§
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Relu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/MatMul¥
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp¢
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/BiasAddq
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_1/Relu§
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_2/MatMul/ReadVariableOp 
dense_2/MatMulMatMuldense_1/Relu:activations:0%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/MatMul¥
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_2/BiasAdd/ReadVariableOp¢
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_2/Relu§
dense_3/MatMul/ReadVariableOpReadVariableOp&dense_3_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_3/MatMul/ReadVariableOp 
dense_3/MatMulMatMuldense_2/Relu:activations:0%dense_3/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/MatMul¥
dense_3/BiasAdd/ReadVariableOpReadVariableOp'dense_3_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_3/BiasAdd/ReadVariableOp¢
dense_3/BiasAddBiasAdddense_3/MatMul:product:0&dense_3/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/BiasAddq
dense_3/ReluReludense_3/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_3/Relu§
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_4/MatMul/ReadVariableOp 
dense_4/MatMulMatMuldense_3/Relu:activations:0%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/MatMul¥
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_4/BiasAdd/ReadVariableOp¢
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/BiasAddq
dense_4/ReluReludense_4/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_4/Relu§
dense_5/MatMul/ReadVariableOpReadVariableOp&dense_5_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_5/MatMul/ReadVariableOp 
dense_5/MatMulMatMuldense_4/Relu:activations:0%dense_5/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/MatMul¥
dense_5/BiasAdd/ReadVariableOpReadVariableOp'dense_5_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_5/BiasAdd/ReadVariableOp¢
dense_5/BiasAddBiasAdddense_5/MatMul:product:0&dense_5/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/BiasAddq
dense_5/ReluReludense_5/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_5/Relu§
dense_6/MatMul/ReadVariableOpReadVariableOp&dense_6_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_6/MatMul/ReadVariableOp 
dense_6/MatMulMatMuldense_5/Relu:activations:0%dense_6/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/MatMul¥
dense_6/BiasAdd/ReadVariableOpReadVariableOp'dense_6_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_6/BiasAdd/ReadVariableOp¢
dense_6/BiasAddBiasAdddense_6/MatMul:product:0&dense_6/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/BiasAddq
dense_6/ReluReludense_6/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_6/Relu§
dense_7/MatMul/ReadVariableOpReadVariableOp&dense_7_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_7/MatMul/ReadVariableOp 
dense_7/MatMulMatMuldense_6/Relu:activations:0%dense_7/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/MatMul¥
dense_7/BiasAdd/ReadVariableOpReadVariableOp'dense_7_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_7/BiasAdd/ReadVariableOp¢
dense_7/BiasAddBiasAdddense_7/MatMul:product:0&dense_7/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/BiasAddq
dense_7/ReluReludense_7/BiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_7/Relu¦
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource*
_output_shapes
:	*
dtype02
dense_8/MatMul/ReadVariableOp
dense_8/MatMulMatMuldense_7/Relu:activations:0%dense_8/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/MatMul¤
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_8/BiasAdd/ReadVariableOp¡
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
dense_8/BiasAddl
IdentityIdentitydense_8/BiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
°
ª
B__inference_dense_6_layer_call_and_return_conditional_losses_12297

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
|
'__inference_dense_4_layer_call_fn_12266

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_107262
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
«
¨
@__inference_dense_layer_call_and_return_conditional_losses_12177

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*.
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ!
Ó
G__inference_functional_1_layer_call_and_return_conditional_losses_11238
input_1
input_2
embedding_model_11107'
#implicit_representation_model_11194'
#implicit_representation_model_11196'
#implicit_representation_model_11198'
#implicit_representation_model_11200'
#implicit_representation_model_11202'
#implicit_representation_model_11204'
#implicit_representation_model_11206'
#implicit_representation_model_11208'
#implicit_representation_model_11210'
#implicit_representation_model_11212'
#implicit_representation_model_11214'
#implicit_representation_model_11216'
#implicit_representation_model_11218'
#implicit_representation_model_11220'
#implicit_representation_model_11222'
#implicit_representation_model_11224'
#implicit_representation_model_11226'
#implicit_representation_model_11228
identity¢'embedding_model/StatefulPartitionedCall¢5implicit_representation_model/StatefulPartitionedCall
'embedding_model/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_model_11107*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_model_layer_call_and_return_conditional_losses_105602)
'embedding_model/StatefulPartitionedCall¡	
5implicit_representation_model/StatefulPartitionedCallStatefulPartitionedCallinput_10embedding_model/StatefulPartitionedCall:output:0#implicit_representation_model_11194#implicit_representation_model_11196#implicit_representation_model_11198#implicit_representation_model_11200#implicit_representation_model_11202#implicit_representation_model_11204#implicit_representation_model_11206#implicit_representation_model_11208#implicit_representation_model_11210#implicit_representation_model_11212#implicit_representation_model_11214#implicit_representation_model_11216#implicit_representation_model_11218#implicit_representation_model_11220#implicit_representation_model_11222#implicit_representation_model_11224#implicit_representation_model_11226#implicit_representation_model_11228*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_1095627
5implicit_representation_model/StatefulPartitionedCall²
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_model_11107*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mulô
IdentityIdentity>implicit_representation_model/StatefulPartitionedCall:output:0(^embedding_model/StatefulPartitionedCall6^implicit_representation_model/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::2R
'embedding_model/StatefulPartitionedCall'embedding_model/StatefulPartitionedCall2n
5implicit_representation_model/StatefulPartitionedCall5implicit_representation_model/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2
ï!
Ó
G__inference_functional_1_layer_call_and_return_conditional_losses_11344

inputs
inputs_1
embedding_model_11297'
#implicit_representation_model_11300'
#implicit_representation_model_11302'
#implicit_representation_model_11304'
#implicit_representation_model_11306'
#implicit_representation_model_11308'
#implicit_representation_model_11310'
#implicit_representation_model_11312'
#implicit_representation_model_11314'
#implicit_representation_model_11316'
#implicit_representation_model_11318'
#implicit_representation_model_11320'
#implicit_representation_model_11322'
#implicit_representation_model_11324'
#implicit_representation_model_11326'
#implicit_representation_model_11328'
#implicit_representation_model_11330'
#implicit_representation_model_11332'
#implicit_representation_model_11334
identity¢'embedding_model/StatefulPartitionedCall¢5implicit_representation_model/StatefulPartitionedCall 
'embedding_model/StatefulPartitionedCallStatefulPartitionedCallinputs_1embedding_model_11297*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_model_layer_call_and_return_conditional_losses_105602)
'embedding_model/StatefulPartitionedCall 	
5implicit_representation_model/StatefulPartitionedCallStatefulPartitionedCallinputs0embedding_model/StatefulPartitionedCall:output:0#implicit_representation_model_11300#implicit_representation_model_11302#implicit_representation_model_11304#implicit_representation_model_11306#implicit_representation_model_11308#implicit_representation_model_11310#implicit_representation_model_11312#implicit_representation_model_11314#implicit_representation_model_11316#implicit_representation_model_11318#implicit_representation_model_11320#implicit_representation_model_11322#implicit_representation_model_11324#implicit_representation_model_11326#implicit_representation_model_11328#implicit_representation_model_11330#implicit_representation_model_11332#implicit_representation_model_11334*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_1095627
5implicit_representation_model/StatefulPartitionedCall²
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_model_11297*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mulô
IdentityIdentity>implicit_representation_model/StatefulPartitionedCall:output:0(^embedding_model/StatefulPartitionedCall6^implicit_representation_model/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::2R
'embedding_model/StatefulPartitionedCall'embedding_model/StatefulPartitionedCall2n
5implicit_representation_model/StatefulPartitionedCall5implicit_representation_model/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ñ!
Ó
G__inference_functional_1_layer_call_and_return_conditional_losses_11289
input_1
input_2
embedding_model_11242'
#implicit_representation_model_11245'
#implicit_representation_model_11247'
#implicit_representation_model_11249'
#implicit_representation_model_11251'
#implicit_representation_model_11253'
#implicit_representation_model_11255'
#implicit_representation_model_11257'
#implicit_representation_model_11259'
#implicit_representation_model_11261'
#implicit_representation_model_11263'
#implicit_representation_model_11265'
#implicit_representation_model_11267'
#implicit_representation_model_11269'
#implicit_representation_model_11271'
#implicit_representation_model_11273'
#implicit_representation_model_11275'
#implicit_representation_model_11277'
#implicit_representation_model_11279
identity¢'embedding_model/StatefulPartitionedCall¢5implicit_representation_model/StatefulPartitionedCall
'embedding_model/StatefulPartitionedCallStatefulPartitionedCallinput_2embedding_model_11242*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_model_layer_call_and_return_conditional_losses_105812)
'embedding_model/StatefulPartitionedCall¡	
5implicit_representation_model/StatefulPartitionedCallStatefulPartitionedCallinput_10embedding_model/StatefulPartitionedCall:output:0#implicit_representation_model_11245#implicit_representation_model_11247#implicit_representation_model_11249#implicit_representation_model_11251#implicit_representation_model_11253#implicit_representation_model_11255#implicit_representation_model_11257#implicit_representation_model_11259#implicit_representation_model_11261#implicit_representation_model_11263#implicit_representation_model_11265#implicit_representation_model_11267#implicit_representation_model_11269#implicit_representation_model_11271#implicit_representation_model_11273#implicit_representation_model_11275#implicit_representation_model_11277#implicit_representation_model_11279*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_1104927
5implicit_representation_model/StatefulPartitionedCall²
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_model_11242*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mulô
IdentityIdentity>implicit_representation_model/StatefulPartitionedCall:output:0(^embedding_model/StatefulPartitionedCall6^implicit_representation_model/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::2R
'embedding_model/StatefulPartitionedCall'embedding_model/StatefulPartitionedCall2n
5implicit_representation_model/StatefulPartitionedCall5implicit_representation_model/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_1:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_2


,__inference_functional_1_layer_call_fn_11806
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17
identity¢StatefulPartitionedCallì
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17* 
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*5
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *P
fKRI
G__inference_functional_1_layer_call_and_return_conditional_losses_114392
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapest
r:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1


=__inference_implicit_representation_model_layer_call_fn_12054
inputs_0
inputs_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16
identity¢StatefulPartitionedCallï
StatefulPartitionedCallStatefulPartitionedCallinputs_0inputs_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*4
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8 *a
f\RZ
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_109562
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Ë
u
/__inference_embedding_model_layer_call_fn_11874

inputs
unknown
identity¢StatefulPartitionedCallð
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *S
fNRL
J__inference_embedding_model_layer_call_and_return_conditional_losses_105812
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_1_layer_call_and_return_conditional_losses_10645

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Á
°
J__inference_embedding_model_layer_call_and_return_conditional_losses_10560

inputs
embedding_matrix_10549
identity¢(embedding_matrix/StatefulPartitionedCall¦
(embedding_matrix/StatefulPartitionedCallStatefulPartitionedCallinputsembedding_matrix_10549*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_embedding_matrix_layer_call_and_return_conditional_losses_104902*
(embedding_matrix/StatefulPartitionedCallþ
reshape/PartitionedCallPartitionedCall1embedding_matrix/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_105142
reshape/PartitionedCall³
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_matrix_10549*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mul
IdentityIdentity reshape/PartitionedCall:output:0)^embedding_matrix/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2T
(embedding_matrix/StatefulPartitionedCall(embedding_matrix/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
2

X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_10956

inputs
inputs_1
dense_10910
dense_10912
dense_1_10915
dense_1_10917
dense_2_10920
dense_2_10922
dense_3_10925
dense_3_10927
dense_4_10930
dense_4_10932
dense_5_10935
dense_5_10937
dense_6_10940
dense_6_10942
dense_7_10945
dense_7_10947
dense_8_10950
dense_8_10952
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCallê
concatenate/PartitionedCallPartitionedCallinputsinputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_105982
concatenate/PartitionedCall¤
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_10910dense_10912*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_106182
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_10915dense_1_10917*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_106452!
dense_1/StatefulPartitionedCall²
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_10920dense_2_10922*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_106722!
dense_2/StatefulPartitionedCall²
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_10925dense_3_10927*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_106992!
dense_3/StatefulPartitionedCall²
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_10930dense_4_10932*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_107262!
dense_4/StatefulPartitionedCall²
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_10935dense_5_10937*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_107532!
dense_5/StatefulPartitionedCall²
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_10940dense_6_10942*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_107802!
dense_6/StatefulPartitionedCall²
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_10945dense_7_10947*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_108072!
dense_7/StatefulPartitionedCall±
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_10950dense_8_10952*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_108332!
dense_8/StatefulPartitionedCall¬
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Î
ª
B__inference_dense_8_layer_call_and_return_conditional_losses_10833

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs

W
+__inference_concatenate_layer_call_fn_12166
inputs_0
inputs_1
identityÔ
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_105982
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:Q M
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/0:QM
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
"
_user_specified_name
inputs/1
Î
ª
B__inference_dense_8_layer_call_and_return_conditional_losses_12336

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddd
IdentityIdentityBiasAdd:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
ß
|
'__inference_dense_7_layer_call_fn_12326

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallö
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_108072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ý
|
'__inference_dense_8_layer_call_fn_12345

inputs
unknown
	unknown_0
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_108332
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
²
p
F__inference_concatenate_layer_call_and_return_conditional_losses_10598

inputs
inputs_1
identity\
concat/axisConst*
_output_shapes
: *
dtype0*
value	B :2
concat/axis
concatConcatV2inputsinputs_1concat/axis:output:0*
N*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
concatc
IdentityIdentityconcat:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*9
_input_shapes(
&:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs:OK
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Û

^
B__inference_reshape_layer_call_and_return_conditional_losses_12137

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2â
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapeo
ReshapeReshapeinputsReshape/shape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
Reshaped
IdentityIdentityReshape:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
°
ª
B__inference_dense_7_layer_call_and_return_conditional_losses_12317

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2	
BiasAddY
ReluReluBiasAdd:output:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2
Relug
IdentityIdentityRelu:activations:0*
T0*(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*/
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:::P L
(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Ü
¹
J__inference_embedding_model_layer_call_and_return_conditional_losses_10543
embedding_input
embedding_matrix_10532
identity¢(embedding_matrix/StatefulPartitionedCall¯
(embedding_matrix/StatefulPartitionedCallStatefulPartitionedCallembedding_inputembedding_matrix_10532*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_embedding_matrix_layer_call_and_return_conditional_losses_104902*
(embedding_matrix/StatefulPartitionedCallþ
reshape/PartitionedCallPartitionedCall1embedding_matrix/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_105142
reshape/PartitionedCall³
,embeddings/Regularizer/Square/ReadVariableOpReadVariableOpembedding_matrix_10532*
_output_shapes

:c*
dtype02.
,embeddings/Regularizer/Square/ReadVariableOp§
embeddings/Regularizer/SquareSquare4embeddings/Regularizer/Square/ReadVariableOp:value:0*
T0*
_output_shapes

:c2
embeddings/Regularizer/Square
embeddings/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
embeddings/Regularizer/Constª
embeddings/Regularizer/SumSum!embeddings/Regularizer/Square:y:0%embeddings/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/Sum
embeddings/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *    2
embeddings/Regularizer/mul/x¬
embeddings/Regularizer/mulMul%embeddings/Regularizer/mul/x:output:0#embeddings/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
embeddings/Regularizer/mul
IdentityIdentity reshape/PartitionedCall:output:0)^embedding_matrix/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:2T
(embedding_matrix/StatefulPartitionedCall(embedding_matrix/StatefulPartitionedCall:X T
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
)
_user_specified_nameembedding_input
2

X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_10901
input_3
input_4
dense_10855
dense_10857
dense_1_10860
dense_1_10862
dense_2_10865
dense_2_10867
dense_3_10870
dense_3_10872
dense_4_10875
dense_4_10877
dense_5_10880
dense_5_10882
dense_6_10885
dense_6_10887
dense_7_10890
dense_7_10892
dense_8_10895
dense_8_10897
identity¢dense/StatefulPartitionedCall¢dense_1/StatefulPartitionedCall¢dense_2/StatefulPartitionedCall¢dense_3/StatefulPartitionedCall¢dense_4/StatefulPartitionedCall¢dense_5/StatefulPartitionedCall¢dense_6/StatefulPartitionedCall¢dense_7/StatefulPartitionedCall¢dense_8/StatefulPartitionedCallê
concatenate/PartitionedCallPartitionedCallinput_3input_4*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *O
fJRH
F__inference_concatenate_layer_call_and_return_conditional_losses_105982
concatenate/PartitionedCall¤
dense/StatefulPartitionedCallStatefulPartitionedCall$concatenate/PartitionedCall:output:0dense_10855dense_10857*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *I
fDRB
@__inference_dense_layer_call_and_return_conditional_losses_106182
dense/StatefulPartitionedCall°
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_10860dense_1_10862*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_1_layer_call_and_return_conditional_losses_106452!
dense_1/StatefulPartitionedCall²
dense_2/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0dense_2_10865dense_2_10867*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_2_layer_call_and_return_conditional_losses_106722!
dense_2/StatefulPartitionedCall²
dense_3/StatefulPartitionedCallStatefulPartitionedCall(dense_2/StatefulPartitionedCall:output:0dense_3_10870dense_3_10872*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_3_layer_call_and_return_conditional_losses_106992!
dense_3/StatefulPartitionedCall²
dense_4/StatefulPartitionedCallStatefulPartitionedCall(dense_3/StatefulPartitionedCall:output:0dense_4_10875dense_4_10877*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_4_layer_call_and_return_conditional_losses_107262!
dense_4/StatefulPartitionedCall²
dense_5/StatefulPartitionedCallStatefulPartitionedCall(dense_4/StatefulPartitionedCall:output:0dense_5_10880dense_5_10882*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_5_layer_call_and_return_conditional_losses_107532!
dense_5/StatefulPartitionedCall²
dense_6/StatefulPartitionedCallStatefulPartitionedCall(dense_5/StatefulPartitionedCall:output:0dense_6_10885dense_6_10887*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_6_layer_call_and_return_conditional_losses_107802!
dense_6/StatefulPartitionedCall²
dense_7/StatefulPartitionedCallStatefulPartitionedCall(dense_6/StatefulPartitionedCall:output:0dense_7_10890dense_7_10892*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_7_layer_call_and_return_conditional_losses_108072!
dense_7/StatefulPartitionedCall±
dense_8/StatefulPartitionedCallStatefulPartitionedCall(dense_7/StatefulPartitionedCall:output:0dense_8_10895dense_8_10897*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_dense_8_layer_call_and_return_conditional_losses_108332!
dense_8/StatefulPartitionedCall¬
IdentityIdentity(dense_8/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall ^dense_2/StatefulPartitionedCall ^dense_3/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall ^dense_5/StatefulPartitionedCall ^dense_6/StatefulPartitionedCall ^dense_7/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0*
_input_shapesp
n:ÿÿÿÿÿÿÿÿÿ:ÿÿÿÿÿÿÿÿÿ::::::::::::::::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2B
dense_2/StatefulPartitionedCalldense_2/StatefulPartitionedCall2B
dense_3/StatefulPartitionedCalldense_3/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall2B
dense_5/StatefulPartitionedCalldense_5/StatefulPartitionedCall2B
dense_6/StatefulPartitionedCalldense_6/StatefulPartitionedCall2B
dense_7/StatefulPartitionedCalldense_7/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:P L
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_3:PL
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
!
_user_specified_name	input_4

C
'__inference_reshape_layer_call_fn_12142

inputs
identityÃ
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8 *K
fFRD
B__inference_reshape_layer_call_and_return_conditional_losses_105142
PartitionedCalll
IdentityIdentityPartitionedCall:output:0*
T0*'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:S O
+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs
Õ
v
0__inference_embedding_matrix_layer_call_fn_12125

inputs
unknown
identity¢StatefulPartitionedCallõ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8 *T
fORM
K__inference_embedding_matrix_layer_call_and_return_conditional_losses_104902
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*+
_output_shapes
:ÿÿÿÿÿÿÿÿÿ2

Identity"
identityIdentity:output:0**
_input_shapes
:ÿÿÿÿÿÿÿÿÿ:22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:ÿÿÿÿÿÿÿÿÿ
 
_user_specified_nameinputs"¸L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*ý
serving_defaulté
;
input_10
serving_default_input_1:0ÿÿÿÿÿÿÿÿÿ
;
input_20
serving_default_input_2:0ÿÿÿÿÿÿÿÿÿQ
implicit_representation_model0
StatefulPartitionedCall:0ÿÿÿÿÿÿÿÿÿtensorflow/serving/predict:Ë
z
layer-0
layer-1
layer_with_weights-0
layer-2
layer_with_weights-1
layer-3
#_self_saveable_object_factories
	optimizer

signatures
trainable_variables
		variables

regularization_losses
	keras_api
ô__call__
õ_default_save_signature
+ö&call_and_return_all_conditional_losses"²w
_tf_keras_networkw{"class_name": "Functional", "name": "functional_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "embedding_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}, "name": "embedding_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_matrix", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 99]}, "dtype": "float32", "input_dim": 99, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": 0.0, "maxval": 1.0, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 99}, "name": "embedding_matrix", "inbound_nodes": [[["embedding_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4]}}, "name": "reshape", "inbound_nodes": [[["embedding_matrix", 0, 0, {}]]]}], "input_layers": [["embedding_input", 0, 0]], "output_layers": [["reshape", 0, 0]]}, "name": "embedding_model", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "implicit_representation_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_8", 0, 0]]}, "name": "implicit_representation_model", "inbound_nodes": [[["input_1", 0, 0, {}], ["embedding_model", 1, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["implicit_representation_model", 1, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 1]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_1", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}, "name": "input_2", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Functional", "config": {"name": "embedding_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}, "name": "embedding_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_matrix", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 99]}, "dtype": "float32", "input_dim": 99, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": 0.0, "maxval": 1.0, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 99}, "name": "embedding_matrix", "inbound_nodes": [[["embedding_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4]}}, "name": "reshape", "inbound_nodes": [[["embedding_matrix", 0, 0, {}]]]}], "input_layers": [["embedding_input", 0, 0]], "output_layers": [["reshape", 0, 0]]}, "name": "embedding_model", "inbound_nodes": [[["input_2", 0, 0, {}]]]}, {"class_name": "Functional", "config": {"name": "implicit_representation_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_8", 0, 0]]}, "name": "implicit_representation_model", "inbound_nodes": [[["input_1", 0, 0, {}], ["embedding_model", 1, 0, {}]]]}], "input_layers": [["input_1", 0, 0], ["input_2", 0, 0]], "output_layers": [["implicit_representation_model", 1, 0]]}}, "training_config": {"loss": "mse", "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}

#_self_saveable_object_factories"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_2", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_2"}}

#_self_saveable_object_factories"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
Î
layer-0
layer_with_weights-0
layer-1
layer-2
#_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
÷__call__
+ø&call_and_return_all_conditional_losses"×
_tf_keras_network»{"class_name": "Functional", "name": "embedding_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "embedding_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}, "name": "embedding_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_matrix", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 99]}, "dtype": "float32", "input_dim": 99, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": 0.0, "maxval": 1.0, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 99}, "name": "embedding_matrix", "inbound_nodes": [[["embedding_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4]}}, "name": "reshape", "inbound_nodes": [[["embedding_matrix", 0, 0, {}]]]}], "input_layers": [["embedding_input", 0, 0]], "output_layers": [["reshape", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "embedding_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}, "name": "embedding_input", "inbound_nodes": []}, {"class_name": "Embedding", "config": {"name": "embedding_matrix", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 99]}, "dtype": "float32", "input_dim": 99, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": 0.0, "maxval": 1.0, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 99}, "name": "embedding_matrix", "inbound_nodes": [[["embedding_input", 0, 0, {}]]]}, {"class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4]}}, "name": "reshape", "inbound_nodes": [[["embedding_matrix", 0, 0, {}]]]}], "input_layers": [["embedding_input", 0, 0]], "output_layers": [["reshape", 0, 0]]}}}
ÆY
layer-0
layer-1
layer-2
layer_with_weights-0
layer-3
layer_with_weights-1
layer-4
layer_with_weights-2
layer-5
layer_with_weights-3
layer-6
layer_with_weights-4
layer-7
layer_with_weights-5
layer-8
layer_with_weights-6
layer-9
 layer_with_weights-7
 layer-10
!layer_with_weights-8
!layer-11
#"_self_saveable_object_factories
#trainable_variables
$	variables
%regularization_losses
&	keras_api
ù__call__
+ú&call_and_return_all_conditional_losses"U
_tf_keras_networkìT{"class_name": "Functional", "name": "implicit_representation_model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "implicit_representation_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_8", 0, 0]]}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 4]}], "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "implicit_representation_model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}, "name": "input_3", "inbound_nodes": []}, {"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}, "name": "input_4", "inbound_nodes": []}, {"class_name": "Concatenate", "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "name": "concatenate", "inbound_nodes": [[["input_3", 0, 0, {}], ["input_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["concatenate", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_2", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_3", "inbound_nodes": [[["dense_2", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_4", "inbound_nodes": [[["dense_3", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_5", "inbound_nodes": [[["dense_4", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_6", "inbound_nodes": [[["dense_5", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_7", "inbound_nodes": [[["dense_6", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_8", "inbound_nodes": [[["dense_7", 0, 0, {}]]]}], "input_layers": [["input_3", 0, 0], ["input_4", 0, 0]], "output_layers": [["dense_8", 0, 0]]}}}
 "
trackable_dict_wrapper
Ï
'iter

(beta_1

)beta_2
	*decay
+learning_rate,mÎ-mÏ.mÐ/mÑ0mÒ1mÓ2mÔ3mÕ4mÖ5m×6mØ7mÙ8mÚ9mÛ:mÜ;mÝ<mÞ=mß>mà,vá-vâ.vã/vä0vå1væ2vç3vè4vé5vê6vë7vì8ví9vî:vï;vð<vñ=vò>vó"
	optimizer
-
ûserving_default"
signature_map
®
,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18"
trackable_list_wrapper
®
,0
-1
.2
/3
04
15
26
37
48
59
610
711
812
913
:14
;15
<16
=17
>18"
trackable_list_wrapper
 "
trackable_list_wrapper
Î

?layers
trainable_variables
@layer_regularization_losses
Alayer_metrics
Bnon_trainable_variables
Cmetrics
		variables

regularization_losses
ô__call__
õ_default_save_signature
+ö&call_and_return_all_conditional_losses
'ö"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper

#D_self_saveable_object_factories"ö
_tf_keras_input_layerÖ{"class_name": "InputLayer", "name": "embedding_input", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "embedding_input"}}
ù
,
embeddings
#E_self_saveable_object_factories
Ftrainable_variables
G	variables
Hregularization_losses
I	keras_api
ü__call__
+ý&call_and_return_all_conditional_losses"³
_tf_keras_layer{"class_name": "Embedding", "name": "embedding_matrix", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, 99]}, "stateful": false, "must_restore_from_config": false, "config": {"name": "embedding_matrix", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 99]}, "dtype": "float32", "input_dim": 99, "output_dim": 4, "embeddings_initializer": {"class_name": "RandomUniform", "config": {"minval": 0.0, "maxval": 1.0, "seed": null}}, "embeddings_regularizer": {"class_name": "L2", "config": {"l2": 0.0}}, "activity_regularizer": null, "embeddings_constraint": null, "mask_zero": false, "input_length": 99}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1]}}

#J_self_saveable_object_factories
Ktrainable_variables
L	variables
Mregularization_losses
N	keras_api
þ__call__
+ÿ&call_and_return_all_conditional_losses"Þ
_tf_keras_layerÄ{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [4]}}}
 "
trackable_dict_wrapper
'
,0"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
°

Olayers
trainable_variables
Player_regularization_losses
Qlayer_metrics
Rnon_trainable_variables
Smetrics
	variables
regularization_losses
÷__call__
+ø&call_and_return_all_conditional_losses
'ø"call_and_return_conditional_losses"
_generic_user_object

#T_self_saveable_object_factories"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_3", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 2]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_3"}}

#U_self_saveable_object_factories"æ
_tf_keras_input_layerÆ{"class_name": "InputLayer", "name": "input_4", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 4]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_4"}}
î
#V_self_saveable_object_factories
Wtrainable_variables
X	variables
Yregularization_losses
Z	keras_api
__call__
+&call_and_return_all_conditional_losses"¸
_tf_keras_layer{"class_name": "Concatenate", "name": "concatenate", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "concatenate", "trainable": true, "dtype": "float32", "axis": -1}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 2]}, {"class_name": "TensorShape", "items": [null, 4]}]}


-kernel
.bias
#[_self_saveable_object_factories
\trainable_variables
]	variables
^regularization_losses
_	keras_api
__call__
+&call_and_return_all_conditional_losses"Æ
_tf_keras_layer¬{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 6}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 6]}}


/kernel
0bias
#`_self_saveable_object_factories
atrainable_variables
b	variables
cregularization_losses
d	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}


1kernel
2bias
#e_self_saveable_object_factories
ftrainable_variables
g	variables
hregularization_losses
i	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}


3kernel
4bias
#j_self_saveable_object_factories
ktrainable_variables
l	variables
mregularization_losses
n	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_3", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}


5kernel
6bias
#o_self_saveable_object_factories
ptrainable_variables
q	variables
rregularization_losses
s	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}


7kernel
8bias
#t_self_saveable_object_factories
utrainable_variables
v	variables
wregularization_losses
x	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_5", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}


9kernel
:bias
#y_self_saveable_object_factories
ztrainable_variables
{	variables
|regularization_losses
}	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_6", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}


;kernel
<bias
#~_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_7", "trainable": true, "dtype": "float32", "units": 128, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}


=kernel
>bias
$_self_saveable_object_factories
trainable_variables
	variables
regularization_losses
	keras_api
__call__
+&call_and_return_all_conditional_losses"Î
_tf_keras_layer´{"class_name": "Dense", "name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 3, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
 "
trackable_dict_wrapper
¦
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17"
trackable_list_wrapper
¦
-0
.1
/2
03
14
25
36
47
58
69
710
811
912
:13
;14
<15
=16
>17"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
#trainable_variables
 layer_regularization_losses
layer_metrics
non_trainable_variables
metrics
$	variables
%regularization_losses
ù__call__
+ú&call_and_return_all_conditional_losses
'ú"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
-:+c2embedding_matrix/embeddings
:	2dense/kernel
:2
dense/bias
": 
2dense_1/kernel
:2dense_1/bias
": 
2dense_2/kernel
:2dense_2/bias
": 
2dense_3/kernel
:2dense_3/bias
": 
2dense_4/kernel
:2dense_4/bias
": 
2dense_5/kernel
:2dense_5/bias
": 
2dense_6/kernel
:2dense_6/bias
": 
2dense_7/kernel
:2dense_7/bias
!:	2dense_8/kernel
:2dense_8/bias
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
(
0"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
'
,0"
trackable_list_wrapper
'
,0"
trackable_list_wrapper
(
0"
trackable_list_wrapper
µ
layers
 layer_regularization_losses
Ftrainable_variables
G	variables
non_trainable_variables
metrics
layer_metrics
Hregularization_losses
ü__call__
+ý&call_and_return_all_conditional_losses
'ý"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
Ktrainable_variables
L	variables
non_trainable_variables
metrics
layer_metrics
Mregularization_losses
þ__call__
+ÿ&call_and_return_all_conditional_losses
'ÿ"call_and_return_conditional_losses"
_generic_user_object
5
0
1
2"
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
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
Wtrainable_variables
X	variables
non_trainable_variables
metrics
layer_metrics
Yregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
-0
.1"
trackable_list_wrapper
.
-0
.1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
layers
 layer_regularization_losses
\trainable_variables
]	variables
non_trainable_variables
 metrics
¡layer_metrics
^regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
/0
01"
trackable_list_wrapper
.
/0
01"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¢layers
 £layer_regularization_losses
atrainable_variables
b	variables
¤non_trainable_variables
¥metrics
¦layer_metrics
cregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
10
21"
trackable_list_wrapper
.
10
21"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
§layers
 ¨layer_regularization_losses
ftrainable_variables
g	variables
©non_trainable_variables
ªmetrics
«layer_metrics
hregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
30
41"
trackable_list_wrapper
.
30
41"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¬layers
 ­layer_regularization_losses
ktrainable_variables
l	variables
®non_trainable_variables
¯metrics
°layer_metrics
mregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
50
61"
trackable_list_wrapper
.
50
61"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
±layers
 ²layer_regularization_losses
ptrainable_variables
q	variables
³non_trainable_variables
´metrics
µlayer_metrics
rregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
70
81"
trackable_list_wrapper
.
70
81"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
¶layers
 ·layer_regularization_losses
utrainable_variables
v	variables
¸non_trainable_variables
¹metrics
ºlayer_metrics
wregularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
90
:1"
trackable_list_wrapper
.
90
:1"
trackable_list_wrapper
 "
trackable_list_wrapper
µ
»layers
 ¼layer_regularization_losses
ztrainable_variables
{	variables
½non_trainable_variables
¾metrics
¿layer_metrics
|regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
;0
<1"
trackable_list_wrapper
.
;0
<1"
trackable_list_wrapper
 "
trackable_list_wrapper
·
Àlayers
 Álayer_regularization_losses
trainable_variables
	variables
Ânon_trainable_variables
Ãmetrics
Älayer_metrics
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
.
=0
>1"
trackable_list_wrapper
.
=0
>1"
trackable_list_wrapper
 "
trackable_list_wrapper
¸
Ålayers
 Ælayer_regularization_losses
trainable_variables
	variables
Çnon_trainable_variables
Èmetrics
Élayer_metrics
regularization_losses
__call__
+&call_and_return_all_conditional_losses
'"call_and_return_conditional_losses"
_generic_user_object
v
0
1
2
3
4
5
6
7
8
9
 10
!11"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
¿

Êtotal

Ëcount
Ì	variables
Í	keras_api"
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
(
0"
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
:  (2total
:  (2count
0
Ê0
Ë1"
trackable_list_wrapper
.
Ì	variables"
_generic_user_object
2:0c2"Adam/embedding_matrix/embeddings/m
$:"	2Adam/dense/kernel/m
:2Adam/dense/bias/m
':%
2Adam/dense_1/kernel/m
 :2Adam/dense_1/bias/m
':%
2Adam/dense_2/kernel/m
 :2Adam/dense_2/bias/m
':%
2Adam/dense_3/kernel/m
 :2Adam/dense_3/bias/m
':%
2Adam/dense_4/kernel/m
 :2Adam/dense_4/bias/m
':%
2Adam/dense_5/kernel/m
 :2Adam/dense_5/bias/m
':%
2Adam/dense_6/kernel/m
 :2Adam/dense_6/bias/m
':%
2Adam/dense_7/kernel/m
 :2Adam/dense_7/bias/m
&:$	2Adam/dense_8/kernel/m
:2Adam/dense_8/bias/m
2:0c2"Adam/embedding_matrix/embeddings/v
$:"	2Adam/dense/kernel/v
:2Adam/dense/bias/v
':%
2Adam/dense_1/kernel/v
 :2Adam/dense_1/bias/v
':%
2Adam/dense_2/kernel/v
 :2Adam/dense_2/bias/v
':%
2Adam/dense_3/kernel/v
 :2Adam/dense_3/bias/v
':%
2Adam/dense_4/kernel/v
 :2Adam/dense_4/bias/v
':%
2Adam/dense_5/kernel/v
 :2Adam/dense_5/bias/v
':%
2Adam/dense_6/kernel/v
 :2Adam/dense_6/bias/v
':%
2Adam/dense_7/kernel/v
 :2Adam/dense_7/bias/v
&:$	2Adam/dense_8/kernel/v
:2Adam/dense_8/bias/v
þ2û
,__inference_functional_1_layer_call_fn_11385
,__inference_functional_1_layer_call_fn_11806
,__inference_functional_1_layer_call_fn_11762
,__inference_functional_1_layer_call_fn_11480À
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
kwonlydefaultsª 
annotationsª *
 
2
 __inference__wrapped_model_10470Þ
²
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *N¢K
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
ê2ç
G__inference_functional_1_layer_call_and_return_conditional_losses_11238
G__inference_functional_1_layer_call_and_return_conditional_losses_11718
G__inference_functional_1_layer_call_and_return_conditional_losses_11289
G__inference_functional_1_layer_call_and_return_conditional_losses_11629À
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
kwonlydefaultsª 
annotationsª *
 
2
/__inference_embedding_model_layer_call_fn_11874
/__inference_embedding_model_layer_call_fn_10586
/__inference_embedding_model_layer_call_fn_10565
/__inference_embedding_model_layer_call_fn_11867À
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
kwonlydefaultsª 
annotationsª *
 
ö2ó
J__inference_embedding_model_layer_call_and_return_conditional_losses_11836
J__inference_embedding_model_layer_call_and_return_conditional_losses_11860
J__inference_embedding_model_layer_call_and_return_conditional_losses_10543
J__inference_embedding_model_layer_call_and_return_conditional_losses_10529À
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
kwonlydefaultsª 
annotationsª *
 
Â2¿
=__inference_implicit_representation_model_layer_call_fn_12054
=__inference_implicit_representation_model_layer_call_fn_11088
=__inference_implicit_representation_model_layer_call_fn_12096
=__inference_implicit_representation_model_layer_call_fn_10995À
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
kwonlydefaultsª 
annotationsª *
 
®2«
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_11943
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_12012
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_10850
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_10901À
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
kwonlydefaultsª 
annotationsª *
 
9B7
#__inference_signature_wrapper_11540input_1input_2
Ú2×
0__inference_embedding_matrix_layer_call_fn_12125¢
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
annotationsª *
 
õ2ò
K__inference_embedding_matrix_layer_call_and_return_conditional_losses_12118¢
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
annotationsª *
 
Ñ2Î
'__inference_reshape_layer_call_fn_12142¢
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
annotationsª *
 
ì2é
B__inference_reshape_layer_call_and_return_conditional_losses_12137¢
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
annotationsª *
 
²2¯
__inference_loss_fn_0_12153
²
FullArgSpec
args 
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsª *¢ 
Õ2Ò
+__inference_concatenate_layer_call_fn_12166¢
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
annotationsª *
 
ð2í
F__inference_concatenate_layer_call_and_return_conditional_losses_12160¢
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
annotationsª *
 
Ï2Ì
%__inference_dense_layer_call_fn_12186¢
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
annotationsª *
 
ê2ç
@__inference_dense_layer_call_and_return_conditional_losses_12177¢
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
annotationsª *
 
Ñ2Î
'__inference_dense_1_layer_call_fn_12206¢
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
annotationsª *
 
ì2é
B__inference_dense_1_layer_call_and_return_conditional_losses_12197¢
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
annotationsª *
 
Ñ2Î
'__inference_dense_2_layer_call_fn_12226¢
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
annotationsª *
 
ì2é
B__inference_dense_2_layer_call_and_return_conditional_losses_12217¢
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
annotationsª *
 
Ñ2Î
'__inference_dense_3_layer_call_fn_12246¢
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
annotationsª *
 
ì2é
B__inference_dense_3_layer_call_and_return_conditional_losses_12237¢
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
annotationsª *
 
Ñ2Î
'__inference_dense_4_layer_call_fn_12266¢
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
annotationsª *
 
ì2é
B__inference_dense_4_layer_call_and_return_conditional_losses_12257¢
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
annotationsª *
 
Ñ2Î
'__inference_dense_5_layer_call_fn_12286¢
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
annotationsª *
 
ì2é
B__inference_dense_5_layer_call_and_return_conditional_losses_12277¢
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
annotationsª *
 
Ñ2Î
'__inference_dense_6_layer_call_fn_12306¢
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
annotationsª *
 
ì2é
B__inference_dense_6_layer_call_and_return_conditional_losses_12297¢
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
annotationsª *
 
Ñ2Î
'__inference_dense_7_layer_call_fn_12326¢
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
annotationsª *
 
ì2é
B__inference_dense_7_layer_call_and_return_conditional_losses_12317¢
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
annotationsª *
 
Ñ2Î
'__inference_dense_8_layer_call_fn_12345¢
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
annotationsª *
 
ì2é
B__inference_dense_8_layer_call_and_return_conditional_losses_12336¢
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
annotationsª *
 ó
 __inference__wrapped_model_10470Î,-./0123456789:;<=>X¢U
N¢K
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
ª "]ªZ
X
implicit_representation_model74
implicit_representation_modelÿÿÿÿÿÿÿÿÿÎ
F__inference_concatenate_layer_call_and_return_conditional_losses_12160Z¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ¥
+__inference_concatenate_layer_call_fn_12166vZ¢W
P¢M
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_1_layer_call_and_return_conditional_losses_12197^/00¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_1_layer_call_fn_12206Q/00¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_2_layer_call_and_return_conditional_losses_12217^120¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_2_layer_call_fn_12226Q120¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_3_layer_call_and_return_conditional_losses_12237^340¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_3_layer_call_fn_12246Q340¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_4_layer_call_and_return_conditional_losses_12257^560¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_4_layer_call_fn_12266Q560¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_5_layer_call_and_return_conditional_losses_12277^780¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_5_layer_call_fn_12286Q780¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_6_layer_call_and_return_conditional_losses_12297^9:0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_6_layer_call_fn_12306Q9:0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¤
B__inference_dense_7_layer_call_and_return_conditional_losses_12317^;<0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 |
'__inference_dense_7_layer_call_fn_12326Q;<0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ£
B__inference_dense_8_layer_call_and_return_conditional_losses_12336]=>0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 {
'__inference_dense_8_layer_call_fn_12345P=>0¢-
&¢#
!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ¡
@__inference_dense_layer_call_and_return_conditional_losses_12177]-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "&¢#

0ÿÿÿÿÿÿÿÿÿ
 y
%__inference_dense_layer_call_fn_12186P-./¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ®
K__inference_embedding_matrix_layer_call_and_return_conditional_losses_12118_,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª ")¢&

0ÿÿÿÿÿÿÿÿÿ
 
0__inference_embedding_matrix_layer_call_fn_12125R,/¢,
%¢"
 
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿº
J__inference_embedding_model_layer_call_and_return_conditional_losses_10529l,@¢=
6¢3
)&
embedding_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 º
J__inference_embedding_model_layer_call_and_return_conditional_losses_10543l,@¢=
6¢3
)&
embedding_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
J__inference_embedding_model_layer_call_and_return_conditional_losses_11836c,7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ±
J__inference_embedding_model_layer_call_and_return_conditional_losses_11860c,7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 
/__inference_embedding_model_layer_call_fn_10565_,@¢=
6¢3
)&
embedding_inputÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_embedding_model_layer_call_fn_10586_,@¢=
6¢3
)&
embedding_inputÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_embedding_model_layer_call_fn_11867V,7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿ
/__inference_embedding_model_layer_call_fn_11874V,7¢4
-¢*
 
inputsÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿê
G__inference_functional_1_layer_call_and_return_conditional_losses_11238,-./0123456789:;<=>`¢]
V¢S
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ê
G__inference_functional_1_layer_call_and_return_conditional_losses_11289,-./0123456789:;<=>`¢]
V¢S
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ì
G__inference_functional_1_layer_call_and_return_conditional_losses_11629 ,-./0123456789:;<=>b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ì
G__inference_functional_1_layer_call_and_return_conditional_losses_11718 ,-./0123456789:;<=>b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Â
,__inference_functional_1_layer_call_fn_11385,-./0123456789:;<=>`¢]
V¢S
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÂ
,__inference_functional_1_layer_call_fn_11480,-./0123456789:;<=>`¢]
V¢S
IF
!
input_1ÿÿÿÿÿÿÿÿÿ
!
input_2ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÄ
,__inference_functional_1_layer_call_fn_11762,-./0123456789:;<=>b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÄ
,__inference_functional_1_layer_call_fn_11806,-./0123456789:;<=>b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿú
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_10850-./0123456789:;<=>`¢]
V¢S
IF
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ú
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_10901-./0123456789:;<=>`¢]
V¢S
IF
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ü
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_11943-./0123456789:;<=>b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 ü
X__inference_implicit_representation_model_layer_call_and_return_conditional_losses_12012-./0123456789:;<=>b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 Ò
=__inference_implicit_representation_model_layer_call_fn_10995-./0123456789:;<=>`¢]
V¢S
IF
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÒ
=__inference_implicit_representation_model_layer_call_fn_11088-./0123456789:;<=>`¢]
V¢S
IF
!
input_3ÿÿÿÿÿÿÿÿÿ
!
input_4ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿÔ
=__inference_implicit_representation_model_layer_call_fn_12054-./0123456789:;<=>b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p

 
ª "ÿÿÿÿÿÿÿÿÿÔ
=__inference_implicit_representation_model_layer_call_fn_12096-./0123456789:;<=>b¢_
X¢U
KH
"
inputs/0ÿÿÿÿÿÿÿÿÿ
"
inputs/1ÿÿÿÿÿÿÿÿÿ
p 

 
ª "ÿÿÿÿÿÿÿÿÿ:
__inference_loss_fn_0_12153,¢

¢ 
ª " ¢
B__inference_reshape_layer_call_and_return_conditional_losses_12137\3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "%¢"

0ÿÿÿÿÿÿÿÿÿ
 z
'__inference_reshape_layer_call_fn_12142O3¢0
)¢&
$!
inputsÿÿÿÿÿÿÿÿÿ
ª "ÿÿÿÿÿÿÿÿÿ
#__inference_signature_wrapper_11540ß,-./0123456789:;<=>i¢f
¢ 
_ª\
,
input_1!
input_1ÿÿÿÿÿÿÿÿÿ
,
input_2!
input_2ÿÿÿÿÿÿÿÿÿ"]ªZ
X
implicit_representation_model74
implicit_representation_modelÿÿÿÿÿÿÿÿÿ