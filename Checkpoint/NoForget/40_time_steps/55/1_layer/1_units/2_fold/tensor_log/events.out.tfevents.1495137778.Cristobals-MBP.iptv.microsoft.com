       ЃK"	  |GжAbrain.Event:2WS}х     yм4	v|GжA"№Ъ
b
lstm_1_inputPlaceholder*
dtype0*
shape: *+
_output_shapes
:џџџџџџџџџ(
l
lstm_1/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
^
lstm_1/random_uniform/minConst*
dtype0*
valueB
 *7П*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
dtype0*
valueB
 *7?*
_output_shapes
: 
Ѕ
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
dtype0*
seed2с№i*
seedБџх)*
T0*
_output_shapes

:
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
T0*
_output_shapes
: 

lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
T0*
_output_shapes

:
{
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes

:

lstm_1/kernel
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
И
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes

:
x
lstm_1/kernel/readIdentitylstm_1/kernel* 
_class
loc:@lstm_1/kernel*
T0*
_output_shapes

:
T
lstm_1/SquareSquarelstm_1/kernel/read*
T0*
_output_shapes

:
Q
lstm_1/mul/xConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
W

lstm_1/mulMullstm_1/mul/xlstm_1/Square*
T0*
_output_shapes

:
]
lstm_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
i

lstm_1/SumSum
lstm_1/mullstm_1/Const*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
Q
lstm_1/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
L

lstm_1/addAddlstm_1/add/x
lstm_1/Sum*
T0*
_output_shapes
: 

%lstm_1/recurrent_kernel/initial_valueConst*
dtype0*)
value B"ыyПФ.QПмП=шж<*
_output_shapes

:

lstm_1/recurrent_kernel
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
ц
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes

:

lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
T0*
_output_shapes

:
[
lstm_1/Const_1Const*
dtype0*
valueB*    *
_output_shapes
:
w
lstm_1/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
Ї
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/Const_1*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes
:
n
lstm_1/bias/readIdentitylstm_1/bias*
_class
loc:@lstm_1/bias*
T0*
_output_shapes
:
k
lstm_1/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
m
lstm_1/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
m
lstm_1/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Њ
lstm_1/strided_sliceStridedSlicelstm_1/kernel/readlstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_1/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
В
lstm_1/strided_slice_1StridedSlicelstm_1/kernel/readlstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_2/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
В
lstm_1/strided_slice_2StridedSlicelstm_1/kernel/readlstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_3/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_3/stack_1Const*
dtype0*
valueB"        *
_output_shapes
:
o
lstm_1/strided_slice_3/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
В
lstm_1/strided_slice_3StridedSlicelstm_1/kernel/readlstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_4/stackConst*
dtype0*
valueB"        *
_output_shapes
:
o
lstm_1/strided_slice_4/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_4/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
М
lstm_1/strided_slice_4StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_5/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
М
lstm_1/strided_slice_5StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_6/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
М
lstm_1/strided_slice_6StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_7/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_7/stack_1Const*
dtype0*
valueB"        *
_output_shapes
:
o
lstm_1/strided_slice_7/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
М
lstm_1/strided_slice_7StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
f
lstm_1/strided_slice_8/stackConst*
dtype0*
valueB: *
_output_shapes
:
h
lstm_1/strided_slice_8/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ќ
lstm_1/strided_slice_8StridedSlicelstm_1/bias/readlstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
f
lstm_1/strided_slice_9/stackConst*
dtype0*
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ќ
lstm_1/strided_slice_9StridedSlicelstm_1/bias/readlstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_10/stackConst*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
А
lstm_1/strided_slice_10StridedSlicelstm_1/bias/readlstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_11/stackConst*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_11/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
i
lstm_1/strided_slice_11/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
А
lstm_1/strided_slice_11StridedSlicelstm_1/bias/readlstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
b
lstm_1/zeros_like	ZerosLikelstm_1_input*
T0*+
_output_shapes
:џџџџџџџџџ(
o
lstm_1/Sum_1/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:

lstm_1/Sum_1Sumlstm_1/zeros_likelstm_1/Sum_1/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
T0*
	keep_dims( *

Tidx0
`
lstm_1/ExpandDims/dimConst*
dtype0*
valueB :
џџџџџџџџџ*
_output_shapes
: 

lstm_1/ExpandDims
ExpandDimslstm_1/Sum_1lstm_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_1/Tile/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:

lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
e
lstm_1/Reshape/shapeConst*
dtype0*
valueB"џџџџ   *
_output_shapes
:
}
lstm_1/ReshapeReshapelstm_1_inputlstm_1/Reshape/shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

lstm_1/MatMulMatMullstm_1/Reshapelstm_1/strided_slice*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
a
lstm_1/stackConst*
dtype0*!
valueB"џџџџ(      *
_output_shapes
:
}
lstm_1/Reshape_1Reshapelstm_1/BiasAddlstm_1/stack*+
_output_shapes
:џџџџџџџџџ(*
T0*
Tshape0
g
lstm_1/Reshape_2/shapeConst*
dtype0*
valueB"џџџџ   *
_output_shapes
:

lstm_1/Reshape_2Reshapelstm_1_inputlstm_1/Reshape_2/shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

lstm_1/MatMul_1MatMullstm_1/Reshape_2lstm_1/strided_slice_1*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
c
lstm_1/stack_1Const*
dtype0*!
valueB"џџџџ(      *
_output_shapes
:

lstm_1/Reshape_3Reshapelstm_1/BiasAdd_1lstm_1/stack_1*+
_output_shapes
:џџџџџџџџџ(*
T0*
Tshape0
g
lstm_1/Reshape_4/shapeConst*
dtype0*
valueB"џџџџ   *
_output_shapes
:

lstm_1/Reshape_4Reshapelstm_1_inputlstm_1/Reshape_4/shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

lstm_1/MatMul_2MatMullstm_1/Reshape_4lstm_1/strided_slice_2*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
c
lstm_1/stack_2Const*
dtype0*!
valueB"џџџџ(      *
_output_shapes
:

lstm_1/Reshape_5Reshapelstm_1/BiasAdd_2lstm_1/stack_2*+
_output_shapes
:џџџџџџџџџ(*
T0*
Tshape0
g
lstm_1/Reshape_6/shapeConst*
dtype0*
valueB"џџџџ   *
_output_shapes
:

lstm_1/Reshape_6Reshapelstm_1_inputlstm_1/Reshape_6/shape*'
_output_shapes
:џџџџџџџџџ*
T0*
Tshape0

lstm_1/MatMul_3MatMullstm_1/Reshape_6lstm_1/strided_slice_3*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
c
lstm_1/stack_3Const*
dtype0*!
valueB"џџџџ(      *
_output_shapes
:

lstm_1/Reshape_7Reshapelstm_1/BiasAdd_3lstm_1/stack_3*+
_output_shapes
:џџџџџџџџџ(*
T0*
Tshape0
T
lstm_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
Р
lstm_1/concatConcatV2lstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7lstm_1/concat/axis*
N*

Tidx0*+
_output_shapes
:џџџџџџџџџ(*
T0
j
lstm_1/transpose/permConst*
dtype0*!
valueB"          *
_output_shapes
:

lstm_1/transpose	Transposelstm_1/concatlstm_1/transpose/perm*
Tperm0*
T0*+
_output_shapes
:(џџџџџџџџџ
\
lstm_1/ShapeShapelstm_1/transpose*
out_type0*
T0*
_output_shapes
:
g
lstm_1/strided_slice_12/stackConst*
dtype0*
valueB: *
_output_shapes
:
i
lstm_1/strided_slice_12/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_12/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ј
lstm_1/strided_slice_12StridedSlicelstm_1/Shapelstm_1/strided_slice_12/stacklstm_1/strided_slice_12/stack_1lstm_1/strided_slice_12/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
g
lstm_1/strided_slice_13/stackConst*
dtype0*
valueB: *
_output_shapes
:
i
lstm_1/strided_slice_13/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_13/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Н
lstm_1/strided_slice_13StridedSlicelstm_1/transposelstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
n
lstm_1/strided_slice_14/stackConst*
dtype0*
valueB"        *
_output_shapes
:
p
lstm_1/strided_slice_14/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_14/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ф
lstm_1/strided_slice_14StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_14/stacklstm_1/strided_slice_14/stack_1lstm_1/strided_slice_14/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_15/stackConst*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ф
lstm_1/strided_slice_15StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_15/stacklstm_1/strided_slice_15/stack_1lstm_1/strided_slice_15/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_16/stackConst*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ф
lstm_1/strided_slice_16StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_16/stacklstm_1/strided_slice_16/stack_1lstm_1/strided_slice_16/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_17/stackConst*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_17/stack_1Const*
dtype0*
valueB"        *
_output_shapes
:
p
lstm_1/strided_slice_17/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ф
lstm_1/strided_slice_17StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_17/stacklstm_1/strided_slice_17/stack_1lstm_1/strided_slice_17/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
S
lstm_1/mul_1/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
lstm_1/mul_1Mullstm_1/Tilelstm_1/mul_1/y*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_4MatMullstm_1/mul_1lstm_1/strided_slice_4*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/add_1Addlstm_1/strided_slice_14lstm_1/MatMul_4*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_2/xConst*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
c
lstm_1/mul_2Mullstm_1/mul_2/xlstm_1/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_2/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
c
lstm_1/add_2Addlstm_1/mul_2lstm_1/add_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_2Const*
dtype0*
valueB
 *    *
_output_shapes
: 
S
lstm_1/Const_3Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_2lstm_1/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_2*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_3/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
lstm_1/mul_3Mullstm_1/Tilelstm_1/mul_3/y*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_5MatMullstm_1/mul_3lstm_1/strided_slice_5*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/add_3Addlstm_1/strided_slice_15lstm_1/MatMul_5*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_4/xConst*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
c
lstm_1/mul_4Mullstm_1/mul_4/xlstm_1/add_3*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_4/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
c
lstm_1/add_4Addlstm_1/mul_4lstm_1/add_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_4Const*
dtype0*
valueB
 *    *
_output_shapes
: 
S
lstm_1/Const_5Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
y
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/add_4lstm_1/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ
j
lstm_1/mul_5Mullstm_1/clip_by_value_1lstm_1/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_6/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
lstm_1/mul_6Mullstm_1/Tilelstm_1/mul_6/y*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_6MatMullstm_1/mul_6lstm_1/strided_slice_6*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/add_5Addlstm_1/strided_slice_16lstm_1/MatMul_6*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/TanhTanhlstm_1/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
h
lstm_1/mul_7Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/add_6Addlstm_1/mul_5lstm_1/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_8/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
lstm_1/mul_8Mullstm_1/Tilelstm_1/mul_8/y*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_7MatMullstm_1/mul_8lstm_1/strided_slice_7*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/add_7Addlstm_1/strided_slice_17lstm_1/MatMul_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_9/xConst*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
c
lstm_1/mul_9Mullstm_1/mul_9/xlstm_1/add_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_8/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
c
lstm_1/add_8Addlstm_1/mul_9lstm_1/add_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_6Const*
dtype0*
valueB
 *    *
_output_shapes
: 
S
lstm_1/Const_7Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
y
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/add_8lstm_1/Const_7*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_6*
T0*'
_output_shapes
:џџџџџџџџџ
U
lstm_1/Tanh_1Tanhlstm_1/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
m
lstm_1/mul_10Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_12*
_output_shapes

::*
dtype0*
dynamic_size( *
clear_after_read(* 
tensor_array_name	output_ta*
element_shape:
Я
lstm_1/TensorArray_1TensorArrayV3lstm_1/strided_slice_12*
_output_shapes

::*
dtype0*
dynamic_size( *
clear_after_read(*
tensor_array_name
input_ta*
element_shape:
o
lstm_1/TensorArrayUnstack/ShapeShapelstm_1/transpose*
out_type0*
T0*
_output_shapes
:
w
-lstm_1/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ћ
'lstm_1/TensorArrayUnstack/strided_sliceStridedSlicelstm_1/TensorArrayUnstack/Shape-lstm_1/TensorArrayUnstack/strided_slice/stack/lstm_1/TensorArrayUnstack/strided_slice/stack_1/lstm_1/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
g
%lstm_1/TensorArrayUnstack/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
g
%lstm_1/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
а
lstm_1/TensorArrayUnstack/rangeRange%lstm_1/TensorArrayUnstack/range/start'lstm_1/TensorArrayUnstack/strided_slice%lstm_1/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ

Alstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm_1/TensorArray_1lstm_1/TensorArrayUnstack/rangelstm_1/transposelstm_1/TensorArray_1:1*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
M
lstm_1/timeConst*
dtype0*
value	B : *
_output_shapes
: 
Ѓ
lstm_1/while/EnterEnterlstm_1/time*
_output_shapes
: *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
А
lstm_1/while/Enter_1Enterlstm_1/TensorArray:1*
_output_shapes
:*
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
Ж
lstm_1/while/Enter_2Enterlstm_1/Tile*'
_output_shapes
:џџџџџџџџџ*
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
Ж
lstm_1/while/Enter_3Enterlstm_1/Tile*'
_output_shapes
:џџџџџџџџџ*
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
w
lstm_1/while/MergeMergelstm_1/while/Enterlstm_1/while/NextIteration*
_output_shapes
: : *
T0*
N

lstm_1/while/Merge_1Mergelstm_1/while/Enter_1lstm_1/while/NextIteration_1*
_output_shapes
:: *
T0*
N

lstm_1/while/Merge_2Mergelstm_1/while/Enter_2lstm_1/while/NextIteration_2*)
_output_shapes
:џџџџџџџџџ: *
T0*
N

lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*)
_output_shapes
:џџџџџџџџџ: *
T0*
N
Д
lstm_1/while/Less/EnterEnterlstm_1/strided_slice_12*
_output_shapes
: *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
g
lstm_1/while/LessLesslstm_1/while/Mergelstm_1/while/Less/Enter*
T0*
_output_shapes
: 
L
lstm_1/while/LoopCondLoopCondlstm_1/while/Less*
_output_shapes
: 

lstm_1/while/SwitchSwitchlstm_1/while/Mergelstm_1/while/LoopCond*%
_class
loc:@lstm_1/while/Merge*
T0*
_output_shapes
: : 

lstm_1/while/Switch_1Switchlstm_1/while/Merge_1lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes

::
К
lstm_1/while/Switch_2Switchlstm_1/while/Merge_2lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_2*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
К
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Y
lstm_1/while/IdentityIdentitylstm_1/while/Switch:1*
T0*
_output_shapes
: 
_
lstm_1/while/Identity_1Identitylstm_1/while/Switch_1:1*
T0*
_output_shapes
:
n
lstm_1/while/Identity_2Identitylstm_1/while/Switch_2:1*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*
T0*'
_output_shapes
:џџџџџџџџџ
ы
$lstm_1/while/TensorArrayReadV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(

&lstm_1/while/TensorArrayReadV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: **

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
ї
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity&lstm_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*'
_output_shapes
:џџџџџџџџџ

 lstm_1/while/strided_slice/stackConst^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:

"lstm_1/while/strided_slice/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

"lstm_1/while/strided_slice/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
з
lstm_1/while/strided_sliceStridedSlicelstm_1/while/TensorArrayReadV3 lstm_1/while/strided_slice/stack"lstm_1/while/strided_slice/stack_1"lstm_1/while/strided_slice/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 

"lstm_1/while/strided_slice_1/stackConst^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_1/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_1/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
п
lstm_1/while/strided_slice_1StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_1/stack$lstm_1/while/strided_slice_1/stack_1$lstm_1/while/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 

"lstm_1/while/strided_slice_2/stackConst^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_2/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_2/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
п
lstm_1/while/strided_slice_2StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_2/stack$lstm_1/while/strided_slice_2/stack_1$lstm_1/while/strided_slice_2/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 

"lstm_1/while/strided_slice_3/stackConst^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_3/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:

$lstm_1/while/strided_slice_3/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
п
lstm_1/while/strided_slice_3StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_3/stack$lstm_1/while/strided_slice_3/stack_1$lstm_1/while/strided_slice_3/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
o
lstm_1/while/mul/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
v
lstm_1/while/mulMullstm_1/while/Identity_2lstm_1/while/mul/y*
T0*'
_output_shapes
:џџџџџџџџџ
Н
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice_4*
_output_shapes

:*
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
Ђ
lstm_1/while/MatMulMatMullstm_1/while/mullstm_1/while/MatMul/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
z
lstm_1/while/addAddlstm_1/while/strided_slicelstm_1/while/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_1/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
s
lstm_1/while/mul_1Mullstm_1/while/mul_1/xlstm_1/while/add*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_1/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *   ?*
_output_shapes
: 
u
lstm_1/while/add_1Addlstm_1/while/mul_1lstm_1/while/add_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/while/ConstConst^lstm_1/while/Identity*
dtype0*
valueB
 *    *
_output_shapes
: 
q
lstm_1/while/Const_1Const^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 

"lstm_1/while/clip_by_value/MinimumMinimumlstm_1/while/add_1lstm_1/while/Const_1*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_2/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
z
lstm_1/while/mul_2Mullstm_1/while/Identity_2lstm_1/while/mul_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
П
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_5*
_output_shapes

:*
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
Ј
lstm_1/while/MatMul_1MatMullstm_1/while/mul_2lstm_1/while/MatMul_1/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/add_2Addlstm_1/while/strided_slice_1lstm_1/while/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_3/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
u
lstm_1/while/mul_3Mullstm_1/while/mul_3/xlstm_1/while/add_2*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_3/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *   ?*
_output_shapes
: 
u
lstm_1/while/add_3Addlstm_1/while/mul_3lstm_1/while/add_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/Const_2Const^lstm_1/while/Identity*
dtype0*
valueB
 *    *
_output_shapes
: 
q
lstm_1/while/Const_3Const^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_3lstm_1/while/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/mul_4Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_3*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_5/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
z
lstm_1/while/mul_5Mullstm_1/while/Identity_2lstm_1/while/mul_5/y*
T0*'
_output_shapes
:џџџџџџџџџ
П
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_6*
_output_shapes

:*
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
Ј
lstm_1/while/MatMul_2MatMullstm_1/while/mul_5lstm_1/while/MatMul_2/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/add_4Addlstm_1/while/strided_slice_2lstm_1/while/MatMul_2*
T0*'
_output_shapes
:џџџџџџџџџ
_
lstm_1/while/TanhTanhlstm_1/while/add_4*
T0*'
_output_shapes
:џџџџџџџџџ
z
lstm_1/while/mul_6Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
s
lstm_1/while/add_5Addlstm_1/while/mul_4lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_7/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
z
lstm_1/while/mul_7Mullstm_1/while/Identity_2lstm_1/while/mul_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
П
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_7*
_output_shapes

:*
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
Ј
lstm_1/while/MatMul_3MatMullstm_1/while/mul_7lstm_1/while/MatMul_3/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/add_6Addlstm_1/while/strided_slice_3lstm_1/while/MatMul_3*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_8/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
u
lstm_1/while/mul_8Mullstm_1/while/mul_8/xlstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_7/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *   ?*
_output_shapes
: 
u
lstm_1/while/add_7Addlstm_1/while/mul_8lstm_1/while/add_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/Const_4Const^lstm_1/while/Identity*
dtype0*
valueB
 *    *
_output_shapes
: 
q
lstm_1/while/Const_5Const^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$lstm_1/while/clip_by_value_2/MinimumMinimumlstm_1/while/add_7lstm_1/while/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
~
lstm_1/while/mul_9Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
љ
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/TensorArray*
is_constant(

0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm_1/while/Identitylstm_1/while/mul_9lstm_1/while/Identity_1*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
: 
n
lstm_1/while/add_8/yConst^lstm_1/while/Identity*
dtype0*
value	B :*
_output_shapes
: 
g
lstm_1/while/add_8Addlstm_1/while/Identitylstm_1/while/add_8/y*
T0*
_output_shapes
: 
`
lstm_1/while/NextIterationNextIterationlstm_1/while/add_8*
T0*
_output_shapes
: 

lstm_1/while/NextIteration_1NextIteration0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
s
lstm_1/while/NextIteration_2NextIterationlstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
O
lstm_1/while/ExitExitlstm_1/while/Switch*
T0*
_output_shapes
: 
U
lstm_1/while/Exit_1Exitlstm_1/while/Switch_1*
T0*
_output_shapes
:
d
lstm_1/while/Exit_2Exitlstm_1/while/Switch_2*
T0*'
_output_shapes
:џџџџџџџџџ
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*
T0*'
_output_shapes
:џџџџџџџџџ
І
)lstm_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm_1/TensorArraylstm_1/while/Exit_1*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/startConst*
dtype0*%
_class
loc:@lstm_1/TensorArray*
value	B : *
_output_shapes
: 

#lstm_1/TensorArrayStack/range/deltaConst*
dtype0*%
_class
loc:@lstm_1/TensorArray*
value	B :*
_output_shapes
: 
ѓ
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*%
_class
loc:@lstm_1/TensorArray*

Tidx0*#
_output_shapes
:џџџџџџџџџ

+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_1*
dtype0*%
_class
loc:@lstm_1/TensorArray*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ*$
element_shape:џџџџџџџџџ
N
lstm_1/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
S

lstm_1/subSublstm_1/while/Exitlstm_1/sub/y*
T0*
_output_shapes
: 
П
lstm_1/TensorArrayReadV3TensorArrayReadV3lstm_1/TensorArray
lstm_1/sublstm_1/while/Exit_1*
dtype0*%
_class
loc:@lstm_1/TensorArray*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/transpose_1/permConst*
dtype0*!
valueB"          *
_output_shapes
:
Б
lstm_1/transpose_1	Transpose+lstm_1/TensorArrayStack/TensorArrayGatherV3lstm_1/transpose_1/perm*
Tperm0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
e
lstm_1/Square_1Squarelstm_1/TensorArrayReadV3*
T0*'
_output_shapes
:џџџџџџџџџ
T
lstm_1/mul_11/xConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
h
lstm_1/mul_11Mullstm_1/mul_11/xlstm_1/Square_1*
T0*'
_output_shapes
:џџџџџџџџџ
_
lstm_1/Const_8Const*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/Sum_2Sumlstm_1/mul_11lstm_1/Const_8*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
S
lstm_1/add_9/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
R
lstm_1/add_9Addlstm_1/add_9/xlstm_1/Sum_2*
T0*
_output_shapes
: 
m
dense_1/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
valueB
 *зГнП*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
dtype0*
valueB
 *зГн?*
_output_shapes
: 
Ј
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2е*
seedБџх)*
T0*
_output_shapes

:
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:

dense_1/kernel
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
М
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:
Z
dense_1/ConstConst*
dtype0*
valueB*    *
_output_shapes
:
x
dense_1/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:
Љ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0*
_output_shapes
:
q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
T0*
_output_shapes
:

dense_1/MatMulMatMullstm_1/TensorArrayReadV3dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:џџџџџџџџџ*
T0*
data_formatNHWC
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
U
lr/initial_valueConst*
dtype0*
valueB
 *
з#<*
_output_shapes
: 
f
lr
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 

	lr/AssignAssignlrlr/initial_value*
validate_shape(*
_class
	loc:@lr*
use_locking(*
T0*
_output_shapes
: 
O
lr/readIdentitylr*
_class
	loc:@lr*
T0*
_output_shapes
: 
V
rho/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: 
g
rho
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 


rho/AssignAssignrhorho/initial_value*
validate_shape(*
_class

loc:@rho*
use_locking(*
T0*
_output_shapes
: 
R
rho/readIdentityrho*
_class

loc:@rho*
T0*
_output_shapes
: 
X
decay/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
decay
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 

decay/AssignAssigndecaydecay/initial_value*
validate_shape(*
_class

loc:@decay*
use_locking(*
T0*
_output_shapes
: 
X

decay/readIdentitydecay*
_class

loc:@decay*
T0*
_output_shapes
: 
]
iterations/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n

iterations
VariableV2*
dtype0*
shape: *
	container *
shared_name *
_output_shapes
: 
Њ
iterations/AssignAssign
iterationsiterations/initial_value*
validate_shape(*
_class
loc:@iterations*
use_locking(*
T0*
_output_shapes
: 
g
iterations/readIdentity
iterations*
_class
loc:@iterations*
T0*
_output_shapes
: 
d
dense_1_sample_weightsPlaceholder*
dtype0*
shape: *#
_output_shapes
:џџџџџџџџџ
i
dense_1_targetPlaceholder*
dtype0*
shape: *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
c
subSubdense_1/Tanhdense_1_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
P
SquareSquaresub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
X
Mean/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
w
MeanMeanSquareMean/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
T0*
	keep_dims( *

Tidx0
[
Mean_1/reduction_indicesConst*
dtype0*
valueB *
_output_shapes
: 
y
Mean_1MeanMeanMean_1/reduction_indices*#
_output_shapes
:џџџџџџџџџ*
T0*
	keep_dims( *

Tidx0
X
mulMulMean_1dense_1_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
O

NotEqual/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
f
NotEqualNotEqualdense_1_sample_weights
NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
S
CastCastNotEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
O
ConstConst*
dtype0*
valueB: *
_output_shapes
:
Y
Mean_2MeanCastConst*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
I
divRealDivmulMean_2*
T0*#
_output_shapes
:џџџџџџџџџ
Q
Const_1Const*
dtype0*
valueB: *
_output_shapes
:
Z
Mean_3MeandivConst_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
L
mul_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
>
mul_1Mulmul_1/xMean_3*
T0*
_output_shapes
: 
>
addAddmul_1
lstm_1/add*
T0*
_output_shapes
: 
@
add_1Addaddlstm_1/add_9*
T0*
_output_shapes
: 


group_depsNoOp^add_1
l
gradients/ShapeConst*
dtype0*
_class

loc:@add_1*
valueB *
_output_shapes
: 
n
gradients/ConstConst*
dtype0*
_class

loc:@add_1*
valueB
 *  ?*
_output_shapes
: 
s
gradients/FillFillgradients/Shapegradients/Const*
_class

loc:@add_1*
T0*
_output_shapes
: 
{
gradients/f_countConst*
dtype0*&
_class
loc:@lstm_1/while/Exit_1*
value	B : *
_output_shapes
: 
в
gradients/f_count_1Entergradients/f_count*
_output_shapes
: **

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 

gradients/MergeMergegradients/f_count_1gradients/NextIteration*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : *
N

gradients/SwitchSwitchgradients/Mergelstm_1/while/LoopCond*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 

gradients/Add/yConst^lstm_1/while/Identity*
dtype0*&
_class
loc:@lstm_1/while/Exit_1*
value	B :*
_output_shapes
: 

gradients/AddAddgradients/Switch:1gradients/Add/y*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
њ+
gradients/NextIterationNextIterationgradients/Add\^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushB^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush_10^gradients/lstm_1/while/mul_9_grad/mul/StackPush2^gradients/lstm_1/while/mul_9_grad/mul_1/StackPushC^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushE^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush_1L^gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushH^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushJ^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush_1T^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush_1B^gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush_10^gradients/lstm_1/while/mul_4_grad/mul/StackPush2^gradients/lstm_1/while/mul_4_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush_10^gradients/lstm_1/while/mul_6_grad/mul/StackPush2^gradients/lstm_1/while/mul_6_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_8_grad/mul/StackPush2^gradients/lstm_1/while/mul_8_grad/mul_1/StackPushC^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushE^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush_1L^gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushA^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushC^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush_1J^gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush_1H^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushJ^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush_1T^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushF^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushH^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush_1R^gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush_1G^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPushI^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_1I^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_2I^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_38^gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushB^gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushG^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPushI^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_1I^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_2I^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_38^gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPushB^gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_7_grad/mul/StackPush2^gradients/lstm_1/while/mul_7_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_3_grad/mul/StackPush2^gradients/lstm_1/while/mul_3_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_1_grad/mul/StackPush2^gradients/lstm_1/while/mul_1_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_5_grad/mul/StackPushB^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush_1@^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush_1G^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPushI^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_1I^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_2I^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_38^gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPushE^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPushG^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_1G^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_2G^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_36^gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPushB^gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_2_grad/mul/StackPush@^gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPush.^gradients/lstm_1/while/mul_grad/mul/StackPush*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
v
gradients/f_count_2Exitgradients/Switch*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
{
gradients/b_countConst*
dtype0*&
_class
loc:@lstm_1/while/Exit_1*
value	B :*
_output_shapes
: 
о
gradients/b_count_1Entergradients/f_count_2*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 

gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : *
N
х
gradients/GreaterEqual/EnterEntergradients/b_count*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant(
 
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
w
gradients/b_count_2LoopCondgradients/GreaterEqual*&
_class
loc:@lstm_1/while/Exit_1*
_output_shapes
: 

gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 

gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
н
gradients/NextIteration_1NextIterationgradients/SubY^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
x
gradients/b_count_3Exitgradients/Switch_1*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
w
gradients/add_1_grad/ShapeConst*
dtype0*
_class

loc:@add_1*
valueB *
_output_shapes
: 
y
gradients/add_1_grad/Shape_1Const*
dtype0*
_class

loc:@add_1*
valueB *
_output_shapes
: 
д
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
_class

loc:@add_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Е
gradients/add_1_grad/SumSumgradients/Fill*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@add_1*
T0*
	keep_dims( *

Tidx0
І
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
_class

loc:@add_1*
T0*
_output_shapes
: *
Tshape0
Й
gradients/add_1_grad/Sum_1Sumgradients/Fill,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@add_1*
T0*
	keep_dims( *

Tidx0
Ќ
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
_class

loc:@add_1*
T0*
_output_shapes
: *
Tshape0
s
gradients/add_grad/ShapeConst*
dtype0*
_class

loc:@add*
valueB *
_output_shapes
: 
u
gradients/add_grad/Shape_1Const*
dtype0*
_class

loc:@add*
valueB *
_output_shapes
: 
Ь
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
_class

loc:@add*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
gradients/add_grad/SumSumgradients/add_1_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@add*
T0*
	keep_dims( *

Tidx0

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
_class

loc:@add*
T0*
_output_shapes
: *
Tshape0
С
gradients/add_grad/Sum_1Sumgradients/add_1_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@add*
T0*
	keep_dims( *

Tidx0
Є
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_class

loc:@add*
T0*
_output_shapes
: *
Tshape0

!gradients/lstm_1/add_9_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/add_9*
valueB *
_output_shapes
: 

#gradients/lstm_1/add_9_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/add_9*
valueB *
_output_shapes
: 
№
1gradients/lstm_1/add_9_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/lstm_1/add_9_grad/Shape#gradients/lstm_1/add_9_grad/Shape_1*
_class
loc:@lstm_1/add_9*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
к
gradients/lstm_1/add_9_grad/SumSumgradients/add_1_grad/Reshape_11gradients/lstm_1/add_9_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@lstm_1/add_9*
T0*
	keep_dims( *

Tidx0
Т
#gradients/lstm_1/add_9_grad/ReshapeReshapegradients/lstm_1/add_9_grad/Sum!gradients/lstm_1/add_9_grad/Shape*
_class
loc:@lstm_1/add_9*
T0*
_output_shapes
: *
Tshape0
о
!gradients/lstm_1/add_9_grad/Sum_1Sumgradients/add_1_grad/Reshape_13gradients/lstm_1/add_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class
loc:@lstm_1/add_9*
T0*
	keep_dims( *

Tidx0
Ш
%gradients/lstm_1/add_9_grad/Reshape_1Reshape!gradients/lstm_1/add_9_grad/Sum_1#gradients/lstm_1/add_9_grad/Shape_1*
_class
loc:@lstm_1/add_9*
T0*
_output_shapes
: *
Tshape0
w
gradients/mul_1_grad/ShapeConst*
dtype0*
_class

loc:@mul_1*
valueB *
_output_shapes
: 
y
gradients/mul_1_grad/Shape_1Const*
dtype0*
_class

loc:@mul_1*
valueB *
_output_shapes
: 
д
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
~
gradients/mul_1_grad/mulMulgradients/add_grad/ReshapeMean_3*
_class

loc:@mul_1*
T0*
_output_shapes
: 
П
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@mul_1*
T0*
	keep_dims( *

Tidx0
І
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
_class

loc:@mul_1*
T0*
_output_shapes
: *
Tshape0

gradients/mul_1_grad/mul_1Mulmul_1/xgradients/add_grad/Reshape*
_class

loc:@mul_1*
T0*
_output_shapes
: 
Х
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@mul_1*
T0*
	keep_dims( *

Tidx0
Ќ
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*
T0*
_output_shapes
: *
Tshape0

gradients/lstm_1/add_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/add*
valueB *
_output_shapes
: 

!gradients/lstm_1/add_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/add*
valueB *
_output_shapes
: 
ш
/gradients/lstm_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/lstm_1/add_grad/Shape!gradients/lstm_1/add_grad/Shape_1*
_class
loc:@lstm_1/add*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
в
gradients/lstm_1/add_grad/SumSumgradients/add_grad/Reshape_1/gradients/lstm_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@lstm_1/add*
T0*
	keep_dims( *

Tidx0
К
!gradients/lstm_1/add_grad/ReshapeReshapegradients/lstm_1/add_grad/Sumgradients/lstm_1/add_grad/Shape*
_class
loc:@lstm_1/add*
T0*
_output_shapes
: *
Tshape0
ж
gradients/lstm_1/add_grad/Sum_1Sumgradients/add_grad/Reshape_11gradients/lstm_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class
loc:@lstm_1/add*
T0*
	keep_dims( *

Tidx0
Р
#gradients/lstm_1/add_grad/Reshape_1Reshapegradients/lstm_1/add_grad/Sum_1!gradients/lstm_1/add_grad/Shape_1*
_class
loc:@lstm_1/add*
T0*
_output_shapes
: *
Tshape0

)gradients/lstm_1/Sum_2_grad/Reshape/shapeConst*
dtype0*
_class
loc:@lstm_1/Sum_2*
valueB"      *
_output_shapes
:
и
#gradients/lstm_1/Sum_2_grad/ReshapeReshape%gradients/lstm_1/add_9_grad/Reshape_1)gradients/lstm_1/Sum_2_grad/Reshape/shape*
_class
loc:@lstm_1/Sum_2*
T0*
_output_shapes

:*
Tshape0

!gradients/lstm_1/Sum_2_grad/ShapeShapelstm_1/mul_11*
out_type0*
T0*
_output_shapes
:*
_class
loc:@lstm_1/Sum_2
е
 gradients/lstm_1/Sum_2_grad/TileTile#gradients/lstm_1/Sum_2_grad/Reshape!gradients/lstm_1/Sum_2_grad/Shape*

Tmultiples0*
_class
loc:@lstm_1/Sum_2*
T0*'
_output_shapes
:џџџџџџџџџ

#gradients/Mean_3_grad/Reshape/shapeConst*
dtype0*
_class
loc:@Mean_3*
valueB:*
_output_shapes
:
Л
gradients/Mean_3_grad/ReshapeReshapegradients/mul_1_grad/Reshape_1#gradients/Mean_3_grad/Reshape/shape*
_class
loc:@Mean_3*
T0*
_output_shapes
:*
Tshape0
y
gradients/Mean_3_grad/ShapeShapediv*
out_type0*
T0*
_output_shapes
:*
_class
loc:@Mean_3
Й
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*

Tmultiples0*
_class
loc:@Mean_3*
T0*#
_output_shapes
:џџџџџџџџџ
{
gradients/Mean_3_grad/Shape_1Shapediv*
out_type0*
T0*
_output_shapes
:*
_class
loc:@Mean_3
{
gradients/Mean_3_grad/Shape_2Const*
dtype0*
_class
loc:@Mean_3*
valueB *
_output_shapes
: 

gradients/Mean_3_grad/ConstConst*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
З
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
_output_shapes
: *
_class
loc:@Mean_3*
T0*
	keep_dims( *

Tidx0

gradients/Mean_3_grad/Const_1Const*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
Л
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*
_output_shapes
: *
_class
loc:@Mean_3*
T0*
	keep_dims( *

Tidx0
|
gradients/Mean_3_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_3*
value	B :*
_output_shapes
: 
Ѓ
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
Ё
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
_class
loc:@Mean_3*
T0*
_output_shapes
: 

gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

DstT0*
_class
loc:@Mean_3*

SrcT0*
_output_shapes
: 
Љ
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
_class
loc:@Mean_3*
T0*#
_output_shapes
:џџџџџџџџџ

'gradients/lstm_1/Sum_grad/Reshape/shapeConst*
dtype0*
_class
loc:@lstm_1/Sum*
valueB"      *
_output_shapes
:
а
!gradients/lstm_1/Sum_grad/ReshapeReshape#gradients/lstm_1/add_grad/Reshape_1'gradients/lstm_1/Sum_grad/Reshape/shape*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes

:*
Tshape0

(gradients/lstm_1/Sum_grad/Tile/multiplesConst*
dtype0*
_class
loc:@lstm_1/Sum*
valueB"      *
_output_shapes
:
Э
gradients/lstm_1/Sum_grad/TileTile!gradients/lstm_1/Sum_grad/Reshape(gradients/lstm_1/Sum_grad/Tile/multiples*

Tmultiples0*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes

:

"gradients/lstm_1/mul_11_grad/ShapeConst*
dtype0* 
_class
loc:@lstm_1/mul_11*
valueB *
_output_shapes
: 

$gradients/lstm_1/mul_11_grad/Shape_1Shapelstm_1/Square_1*
out_type0*
T0*
_output_shapes
:* 
_class
loc:@lstm_1/mul_11
є
2gradients/lstm_1/mul_11_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/lstm_1/mul_11_grad/Shape$gradients/lstm_1/mul_11_grad/Shape_1* 
_class
loc:@lstm_1/mul_11*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
 gradients/lstm_1/mul_11_grad/mulMul gradients/lstm_1/Sum_2_grad/Tilelstm_1/Square_1* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:џџџџџџџџџ
п
 gradients/lstm_1/mul_11_grad/SumSum gradients/lstm_1/mul_11_grad/mul2gradients/lstm_1/mul_11_grad/BroadcastGradientArgs*
_output_shapes
:* 
_class
loc:@lstm_1/mul_11*
T0*
	keep_dims( *

Tidx0
Ц
$gradients/lstm_1/mul_11_grad/ReshapeReshape gradients/lstm_1/mul_11_grad/Sum"gradients/lstm_1/mul_11_grad/Shape* 
_class
loc:@lstm_1/mul_11*
T0*
_output_shapes
: *
Tshape0
А
"gradients/lstm_1/mul_11_grad/mul_1Mullstm_1/mul_11/x gradients/lstm_1/Sum_2_grad/Tile* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:џџџџџџџџџ
х
"gradients/lstm_1/mul_11_grad/Sum_1Sum"gradients/lstm_1/mul_11_grad/mul_14gradients/lstm_1/mul_11_grad/BroadcastGradientArgs:1*
_output_shapes
:* 
_class
loc:@lstm_1/mul_11*
T0*
	keep_dims( *

Tidx0
н
&gradients/lstm_1/mul_11_grad/Reshape_1Reshape"gradients/lstm_1/mul_11_grad/Sum_1$gradients/lstm_1/mul_11_grad/Shape_1* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
s
gradients/div_grad/ShapeShapemul*
out_type0*
T0*
_output_shapes
:*
_class

loc:@div
u
gradients/div_grad/Shape_1Const*
dtype0*
_class

loc:@div*
valueB *
_output_shapes
: 
Ь
(gradients/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_grad/Shapegradients/div_grad/Shape_1*
_class

loc:@div*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/div_grad/RealDivRealDivgradients/Mean_3_grad/truedivMean_2*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ
Л
gradients/div_grad/SumSumgradients/div_grad/RealDiv(gradients/div_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@div*
T0*
	keep_dims( *

Tidx0
Ћ
gradients/div_grad/ReshapeReshapegradients/div_grad/Sumgradients/div_grad/Shape*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ*
Tshape0
h
gradients/div_grad/NegNegmul*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/div_grad/RealDiv_1RealDivgradients/div_grad/NegMean_2*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/div_grad/RealDiv_2RealDivgradients/div_grad/RealDiv_1Mean_2*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ
 
gradients/div_grad/mulMulgradients/Mean_3_grad/truedivgradients/div_grad/RealDiv_2*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ
Л
gradients/div_grad/Sum_1Sumgradients/div_grad/mul*gradients/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@div*
T0*
	keep_dims( *

Tidx0
Є
gradients/div_grad/Reshape_1Reshapegradients/div_grad/Sum_1gradients/div_grad/Shape_1*
_class

loc:@div*
T0*
_output_shapes
: *
Tshape0

gradients/lstm_1/mul_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/mul*
valueB *
_output_shapes
: 

!gradients/lstm_1/mul_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/mul*
valueB"      *
_output_shapes
:
ш
/gradients/lstm_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/lstm_1/mul_grad/Shape!gradients/lstm_1/mul_grad/Shape_1*
_class
loc:@lstm_1/mul*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/lstm_1/mul_grad/mulMulgradients/lstm_1/Sum_grad/Tilelstm_1/Square*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:
г
gradients/lstm_1/mul_grad/SumSumgradients/lstm_1/mul_grad/mul/gradients/lstm_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@lstm_1/mul*
T0*
	keep_dims( *

Tidx0
К
!gradients/lstm_1/mul_grad/ReshapeReshapegradients/lstm_1/mul_grad/Sumgradients/lstm_1/mul_grad/Shape*
_class
loc:@lstm_1/mul*
T0*
_output_shapes
: *
Tshape0

gradients/lstm_1/mul_grad/mul_1Mullstm_1/mul/xgradients/lstm_1/Sum_grad/Tile*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:
й
gradients/lstm_1/mul_grad/Sum_1Sumgradients/lstm_1/mul_grad/mul_11gradients/lstm_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class
loc:@lstm_1/mul*
T0*
	keep_dims( *

Tidx0
Ш
#gradients/lstm_1/mul_grad/Reshape_1Reshapegradients/lstm_1/mul_grad/Sum_1!gradients/lstm_1/mul_grad/Shape_1*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:*
Tshape0
Ж
$gradients/lstm_1/Square_1_grad/mul/xConst'^gradients/lstm_1/mul_11_grad/Reshape_1*
dtype0*"
_class
loc:@lstm_1/Square_1*
valueB
 *   @*
_output_shapes
: 
П
"gradients/lstm_1/Square_1_grad/mulMul$gradients/lstm_1/Square_1_grad/mul/xlstm_1/TensorArrayReadV3*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:џџџџџџџџџ
Э
$gradients/lstm_1/Square_1_grad/mul_1Mul&gradients/lstm_1/mul_11_grad/Reshape_1"gradients/lstm_1/Square_1_grad/mul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:џџџџџџџџџ
v
gradients/mul_grad/ShapeShapeMean_1*
out_type0*
T0*
_output_shapes
:*
_class

loc:@mul

gradients/mul_grad/Shape_1Shapedense_1_sample_weights*
out_type0*
T0*
_output_shapes
:*
_class

loc:@mul
Ь
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
_class

loc:@mul*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/mul_grad/mulMulgradients/div_grad/Reshapedense_1_sample_weights*
_class

loc:@mul*
T0*#
_output_shapes
:џџџџџџџџџ
З
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@mul*
T0*
	keep_dims( *

Tidx0
Ћ
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
_class

loc:@mul*
T0*#
_output_shapes
:џџџџџџџџџ*
Tshape0

gradients/mul_grad/mul_1MulMean_1gradients/div_grad/Reshape*
_class

loc:@mul*
T0*#
_output_shapes
:џџџџџџџџџ
Н
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@mul*
T0*
	keep_dims( *

Tidx0
Б
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
_class

loc:@mul*
T0*#
_output_shapes
:џџџџџџџџџ*
Tshape0
Џ
"gradients/lstm_1/Square_grad/mul/xConst$^gradients/lstm_1/mul_grad/Reshape_1*
dtype0* 
_class
loc:@lstm_1/Square*
valueB
 *   @*
_output_shapes
: 
Њ
 gradients/lstm_1/Square_grad/mulMul"gradients/lstm_1/Square_grad/mul/xlstm_1/kernel/read* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:
Л
"gradients/lstm_1/Square_grad/mul_1Mul#gradients/lstm_1/mul_grad/Reshape_1 gradients/lstm_1/Square_grad/mul* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:
z
gradients/Mean_1_grad/ShapeShapeMean*
out_type0*
T0*
_output_shapes
:*
_class
loc:@Mean_1
w
gradients/Mean_1_grad/SizeConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 

gradients/Mean_1_grad/addAddMean_1/reduction_indicesgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
: 

gradients/Mean_1_grad/modFloorModgradients/Mean_1_grad/addgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
: 

gradients/Mean_1_grad/Shape_1Const*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
~
!gradients/Mean_1_grad/range/startConst*
dtype0*
_class
loc:@Mean_1*
value	B : *
_output_shapes
: 
~
!gradients/Mean_1_grad/range/deltaConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
Щ
gradients/Mean_1_grad/rangeRange!gradients/Mean_1_grad/range/startgradients/Mean_1_grad/Size!gradients/Mean_1_grad/range/delta*
_class
loc:@Mean_1*

Tidx0*
_output_shapes
:
}
 gradients/Mean_1_grad/Fill/valueConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
Ё
gradients/Mean_1_grad/FillFillgradients/Mean_1_grad/Shape_1 gradients/Mean_1_grad/Fill/value*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
ї
#gradients/Mean_1_grad/DynamicStitchDynamicStitchgradients/Mean_1_grad/rangegradients/Mean_1_grad/modgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Fill*
_class
loc:@Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ*
N
|
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
З
gradients/Mean_1_grad/MaximumMaximum#gradients/Mean_1_grad/DynamicStitchgradients/Mean_1_grad/Maximum/y*
_class
loc:@Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
Џ
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Maximum*
_class
loc:@Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
Е
gradients/Mean_1_grad/ReshapeReshapegradients/mul_grad/Reshape#gradients/Mean_1_grad/DynamicStitch*
_class
loc:@Mean_1*
T0*
_output_shapes
:*
Tshape0
Б
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/floordiv*

Tmultiples0*
_class
loc:@Mean_1*
T0*
_output_shapes
:
|
gradients/Mean_1_grad/Shape_2ShapeMean*
out_type0*
T0*
_output_shapes
:*
_class
loc:@Mean_1
~
gradients/Mean_1_grad/Shape_3ShapeMean_1*
out_type0*
T0*
_output_shapes
:*
_class
loc:@Mean_1

gradients/Mean_1_grad/ConstConst*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
З
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const*
_output_shapes
: *
_class
loc:@Mean_1*
T0*
	keep_dims( *

Tidx0

gradients/Mean_1_grad/Const_1Const*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
Л
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_3gradients/Mean_1_grad/Const_1*
_output_shapes
: *
_class
loc:@Mean_1*
T0*
	keep_dims( *

Tidx0
~
!gradients/Mean_1_grad/Maximum_1/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
Ї
gradients/Mean_1_grad/Maximum_1Maximumgradients/Mean_1_grad/Prod_1!gradients/Mean_1_grad/Maximum_1/y*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
Ѕ
 gradients/Mean_1_grad/floordiv_1FloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum_1*
_class
loc:@Mean_1*
T0*
_output_shapes
: 

gradients/Mean_1_grad/CastCast gradients/Mean_1_grad/floordiv_1*

DstT0*
_class
loc:@Mean_1*

SrcT0*
_output_shapes
: 
Љ
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
_class
loc:@Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
x
gradients/Mean_grad/ShapeShapeSquare*
out_type0*
T0*
_output_shapes
:*
_class
	loc:@Mean
s
gradients/Mean_grad/SizeConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 

gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 
w
gradients/Mean_grad/Shape_1Const*
dtype0*
_class
	loc:@Mean*
valueB *
_output_shapes
: 
z
gradients/Mean_grad/range/startConst*
dtype0*
_class
	loc:@Mean*
value	B : *
_output_shapes
: 
z
gradients/Mean_grad/range/deltaConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
П
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*
_class
	loc:@Mean*

Tidx0*
_output_shapes
:
y
gradients/Mean_grad/Fill/valueConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 

gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
_class
	loc:@Mean*
T0*
_output_shapes
: 
ы
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
_class
	loc:@Mean*
T0*#
_output_shapes
:џџџџџџџџџ*
N
x
gradients/Mean_grad/Maximum/yConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
Џ
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
_class
	loc:@Mean*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*
T0*
_output_shapes
:
В
gradients/Mean_grad/ReshapeReshapegradients/Mean_1_grad/truediv!gradients/Mean_grad/DynamicStitch*
_class
	loc:@Mean*
T0*
_output_shapes
:*
Tshape0
С
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
_class
	loc:@Mean*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
z
gradients/Mean_grad/Shape_2ShapeSquare*
out_type0*
T0*
_output_shapes
:*
_class
	loc:@Mean
x
gradients/Mean_grad/Shape_3ShapeMean*
out_type0*
T0*
_output_shapes
:*
_class
	loc:@Mean
|
gradients/Mean_grad/ConstConst*
dtype0*
_class
	loc:@Mean*
valueB: *
_output_shapes
:
Џ
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
_output_shapes
: *
_class
	loc:@Mean*
T0*
	keep_dims( *

Tidx0
~
gradients/Mean_grad/Const_1Const*
dtype0*
_class
	loc:@Mean*
valueB: *
_output_shapes
:
Г
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_output_shapes
: *
_class
	loc:@Mean*
T0*
	keep_dims( *

Tidx0
z
gradients/Mean_grad/Maximum_1/yConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_class
	loc:@Mean*
T0*
_output_shapes
: 

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
T0*
_output_shapes
: 

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
_class
	loc:@Mean*

SrcT0*
_output_shapes
: 
Ў
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_class
	loc:@Mean*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

gradients/Square_grad/mul/xConst^gradients/Mean_grad/truediv*
dtype0*
_class
loc:@Square*
valueB
 *   @*
_output_shapes
: 

gradients/Square_grad/mulMulgradients/Square_grad/mul/xsub*
_class
loc:@Square*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
А
gradients/Square_grad/mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/mul*
_class
loc:@Square*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
|
gradients/sub_grad/ShapeShapedense_1/Tanh*
out_type0*
T0*
_output_shapes
:*
_class

loc:@sub

gradients/sub_grad/Shape_1Shapedense_1_target*
out_type0*
T0*
_output_shapes
:*
_class

loc:@sub
Ь
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
_class

loc:@sub*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
М
gradients/sub_grad/SumSumgradients/Square_grad/mul_1(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@sub*
T0*
	keep_dims( *

Tidx0
Џ
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
_class

loc:@sub*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Р
gradients/sub_grad/Sum_1Sumgradients/Square_grad/mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@sub*
T0*
	keep_dims( *

Tidx0
r
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_class

loc:@sub*
T0*
_output_shapes
:
М
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
_class

loc:@sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ*
Tshape0
­
$gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhgradients/sub_grad/Reshape*
_class
loc:@dense_1/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
П
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_1/Tanh_grad/TanhGrad*"
_class
loc:@dense_1/BiasAdd*
T0*
_output_shapes
:*
data_formatNHWC
ф
$gradients/dense_1/MatMul_grad/MatMulMatMul$gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
transpose_b(*
transpose_a( *!
_class
loc:@dense_1/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
т
&gradients/dense_1/MatMul_grad/MatMul_1MatMullstm_1/TensorArrayReadV3$gradients/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_1/MatMul*
T0*
_output_shapes

:
С
gradients/AddNAddN$gradients/lstm_1/Square_1_grad/mul_1$gradients/dense_1/MatMul_grad/MatMul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:џџџџџџџџџ*
N
х
Igradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArraylstm_1/while/Exit_1*
source	gradients*%
_class
loc:@lstm_1/TensorArray*
_output_shapes

::
ќ
Egradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitylstm_1/while/Exit_1J^gradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
:
з
Kgradients/lstm_1/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Igradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3
lstm_1/subgradients/AddNEgradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
: 
h
gradients/zeros_like	ZerosLikelstm_1/while/Exit_2*
T0*'
_output_shapes
:џџџџџџџџџ
j
gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_3*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
)gradients/lstm_1/while/Exit_1_grad/b_exitEnterKgradients/lstm_1/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 

)gradients/lstm_1/while/Exit_2_grad/b_exitEntergradients/zeros_like*'
_output_shapes
:џџџџџџџџџ*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_2*
is_constant( 

)gradients/lstm_1/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*'
_output_shapes
:џџџџџџџџџ*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_3*
is_constant( 
ь
-gradients/lstm_1/while/Switch_1_grad/b_switchMerge)gradients/lstm_1/while/Exit_1_grad/b_exit4gradients/lstm_1/while/Switch_1_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: : *
N
§
-gradients/lstm_1/while/Switch_2_grad/b_switchMerge)gradients/lstm_1/while/Exit_2_grad/b_exit4gradients/lstm_1/while/Switch_2_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_2*
T0*)
_output_shapes
:џџџџџџџџџ: *
N
§
-gradients/lstm_1/while/Switch_3_grad/b_switchMerge)gradients/lstm_1/while/Exit_3_grad/b_exit4gradients/lstm_1/while/Switch_3_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_3*
T0*)
_output_shapes
:џџџџџџџџџ: *
N
Ф
*gradients/lstm_1/while/Merge_1_grad/SwitchSwitch-gradients/lstm_1/while/Switch_1_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: : 
ц
*gradients/lstm_1/while/Merge_2_grad/SwitchSwitch-gradients/lstm_1/while/Switch_2_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_2*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
ц
*gradients/lstm_1/while/Merge_3_grad/SwitchSwitch-gradients/lstm_1/while/Switch_3_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
І
(gradients/lstm_1/while/Enter_1_grad/ExitExit*gradients/lstm_1/while/Merge_1_grad/Switch*'
_class
loc:@lstm_1/while/Enter_1*
T0*
_output_shapes
: 
Д
ggradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/TensorArray*
is_constant(
ы
agradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3ggradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter,gradients/lstm_1/while/Merge_1_grad/Switch:1*
source	gradients*%
_class
loc:@lstm_1/TensorArray*
_output_shapes

::
У
]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity,gradients/lstm_1/while/Merge_1_grad/Switch:1b^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
: 
т
Wgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
	elem_type0*
_output_shapes
:

Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnterRefEnterWgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
is_constant(
ф
[gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPush	StackPushZgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnterlstm_1/while/Identity^gradients/Add*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
T0*
swap_memory(*
_output_shapes
:

cgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop/RefEnterRefEnterWgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
is_constant(
Ч
Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopStackPopcgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
	elem_type0*
_output_shapes
: 
Й+
Xgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger[^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopA^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_9_grad/mul/StackPop1^gradients/lstm_1/while/mul_9_grad/mul_1/StackPopB^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopD^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1K^gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopG^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopI^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1S^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1A^gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_4_grad/mul/StackPop1^gradients/lstm_1/while/mul_4_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_6_grad/mul/StackPop1^gradients/lstm_1/while/mul_6_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_8_grad/mul/StackPop1^gradients/lstm_1/while/mul_8_grad/mul_1/StackPopB^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopD^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1K^gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop@^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopB^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1I^gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1G^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopI^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1S^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopE^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopG^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1Q^gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1F^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopA^gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopF^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPopA^gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_7_grad/mul/StackPop1^gradients/lstm_1/while/mul_7_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_3_grad/mul/StackPop1^gradients/lstm_1/while/mul_3_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_1_grad/mul/StackPop1^gradients/lstm_1/while/mul_1_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_5_grad/mul/StackPopA^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1?^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1F^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPopD^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopF^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1F^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2F^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_35^gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPopA^gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_2_grad/mul/StackPop?^gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop-^gradients/lstm_1/while/mul_grad/mul/StackPop*%
_class
loc:@lstm_1/TensorArray
с
Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3agradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*%
_class
loc:@lstm_1/TensorArray*'
_output_shapes
:џџџџџџџџџ
§
gradients/AddN_1AddN,gradients/lstm_1/while/Merge_2_grad/Switch:1Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:џџџџџџџџџ*
N
Њ
'gradients/lstm_1/while/mul_9_grad/ShapeShapelstm_1/while/clip_by_value_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_9
Ѓ
)gradients/lstm_1/while/mul_9_grad/Shape_1Shapelstm_1/while/Tanh_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_9
Ќ
=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
І
Agradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_9_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
ћ
@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
Ќ
Cgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_9_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
џ
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_9*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Д
+gradients/lstm_1/while/mul_9_grad/mul/f_accStack*

stack_name *?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Ї
.gradients/lstm_1/while/mul_9_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_9_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
is_constant(

/gradients/lstm_1/while/mul_9_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_9_grad/mul/RefEnterlstm_1/while/Tanh_1^gradients/Add*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
К
7gradients/lstm_1/while/mul_9_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_9_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
is_constant(
ў
.gradients/lstm_1/while/mul_9_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_9_grad/mul/StackPop/RefEnter^gradients/Sub*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
Ч
%gradients/lstm_1/while/mul_9_grad/mulMulgradients/AddN_1.gradients/lstm_1/while/mul_9_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_9_grad/SumSum%gradients/lstm_1/while/mul_9_grad/mul7gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_9*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/mul_9_grad/ReshapeReshape%gradients/lstm_1/while/mul_9_grad/Sum@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
П
-gradients/lstm_1/while/mul_9_grad/mul_1/f_accStack*

stack_name *H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Д
0gradients/lstm_1/while/mul_9_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_9_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
is_constant(

1gradients/lstm_1/while/mul_9_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_9_grad/mul_1/RefEnterlstm_1/while/clip_by_value_2^gradients/Add*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
Ч
9gradients/lstm_1/while/mul_9_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_9_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
is_constant(

0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_9_grad/mul_1/StackPop/RefEnter^gradients/Sub*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
Ы
'gradients/lstm_1/while/mul_9_grad/mul_1Mul0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopgradients/AddN_1*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_9_grad/Sum_1Sum'gradients/lstm_1/while/mul_9_grad/mul_19gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_9*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/mul_9_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_9_grad/Sum_1Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ц
1gradients/lstm_1/while/clip_by_value_2_grad/ShapeShape$lstm_1/while/clip_by_value_2/Minimum*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
З
3gradients/lstm_1/while/clip_by_value_2_grad/Shape_1Const^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB *
_output_shapes
: 
Э
3gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Shape)gradients/lstm_1/while/mul_9_grad/Reshape*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
Н
7gradients/lstm_1/while/clip_by_value_2_grad/zeros/ConstConst^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
_output_shapes
: 

1gradients/lstm_1/while/clip_by_value_2_grad/zerosFill3gradients/lstm_1/while/clip_by_value_2_grad/Shape_27gradients/lstm_1/while/clip_by_value_2_grad/zeros/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ
т
>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_accStack*

stack_name *Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
ш
Agradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
к
Bgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush	StackPushAgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter$lstm_1/while/clip_by_value_2/Minimum^gradients/Add*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Jgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
П
Agradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopStackPopJgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
д
@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1Stack*

stack_name *J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
м
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1RefEnter@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
is_constant(
О
Dgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush_1	StackPushCgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1lstm_1/while/Const_4^gradients/Add*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
T0*
swap_memory(*
_output_shapes
:
я
Lgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1/RefEnterRefEnter@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
is_constant(
Ђ
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1StackPopLgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
: 
Г
8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqualGreaterEqualAgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopCgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ
Р
Ggradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
Я
Jgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
is_constant(
Ю
Kgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPush	StackPushJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnter1gradients/lstm_1/while/clip_by_value_2_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
swap_memory(*
_output_shapes
:
т
Sgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
is_constant(

Jgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopStackPopSgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
Щ
Agradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop3gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
2gradients/lstm_1/while/clip_by_value_2_grad/SelectSelect8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ
и
6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ
З
4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/lstm_1/while/clip_by_value_2_grad/SumSum2gradients/lstm_1/while/clip_by_value_2_grad/SelectAgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
	keep_dims( *

Tidx0
Ќ
3gradients/lstm_1/while/clip_by_value_2_grad/ReshapeReshape/gradients/lstm_1/while/clip_by_value_2_grad/SumJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Є
1gradients/lstm_1/while/clip_by_value_2_grad/Sum_1Sum4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Cgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
	keep_dims( *

Tidx0

5gradients/lstm_1/while/clip_by_value_2_grad/Reshape_1Reshape1gradients/lstm_1/while/clip_by_value_2_grad/Sum_13gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
_output_shapes
: *
Tshape0
ю
+gradients/lstm_1/while/Tanh_1_grad/TanhGradTanhGrad.gradients/lstm_1/while/mul_9_grad/mul/StackPop+gradients/lstm_1/while/mul_9_grad/Reshape_1*&
_class
loc:@lstm_1/while/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
Н
4gradients/lstm_1/while/Switch_1_grad_1/NextIterationNextIteration,gradients/lstm_1/while/Merge_1_grad/Switch:1*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: 
Ф
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ShapeShapelstm_1/while/add_7*
out_type0*
T0*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
Ч
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB *
_output_shapes
: 
ч
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2Shape3gradients/lstm_1/while/clip_by_value_2_grad/Reshape*
out_type0*
T0*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
Э
?gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *    *
_output_shapes
: 
Њ
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosFill;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2?gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
н
Cgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_accStack*

stack_name *P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
ш
Fgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
Ш
Ggradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush	StackPushFgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_7^gradients/Add*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Ogradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
П
Fgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopStackPopOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
с
Egradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
ю
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1RefEnterEgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
а
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush_1	StackPushHgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_5^gradients/Add*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:

Qgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
Д
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1StackPopQgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
: 
Ч
=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual	LessEqualFgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopHgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
а
Ogradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
ч
Rgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
ю
Sgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnter9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape^gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
њ
[gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
Б
Rgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopStackPop[gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
щ
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
:gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectSelect=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
э
>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ
с
<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
О
7gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumSum:gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectIgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
	keep_dims( *

Tidx0
Ь
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeReshape7gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ф
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1Sum<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Kgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
	keep_dims( *

Tidx0
Ј
=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape_1Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
_output_shapes
: *
Tshape0
з
gradients/AddN_2AddN,gradients/lstm_1/while/Merge_3_grad/Switch:1+gradients/lstm_1/while/Tanh_1_grad/TanhGrad*'
_class
loc:@lstm_1/while/Merge_3*
T0*'
_output_shapes
:џџџџџџџџџ*
N
 
'gradients/lstm_1/while/add_5_grad/ShapeShapelstm_1/while/mul_4*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_5
Ђ
)gradients/lstm_1/while/add_5_grad/Shape_1Shapelstm_1/while/mul_6*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_5
Ќ
=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
І
Agradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_5_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_5*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
ћ
@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
Ќ
Cgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_5_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_5*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
џ
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_5*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
%gradients/lstm_1/while/add_5_grad/SumSumgradients/AddN_27gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_5*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/add_5_grad/ReshapeReshape%gradients/lstm_1/while/add_5_grad/Sum@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
т
'gradients/lstm_1/while/add_5_grad/Sum_1Sumgradients/AddN_29gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_5*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/add_5_grad/Reshape_1Reshape'gradients/lstm_1/while/add_5_grad/Sum_1Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
 
'gradients/lstm_1/while/add_7_grad/ShapeShapelstm_1/while/mul_8*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_7
Ѓ
)gradients/lstm_1/while/add_7_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_7*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_7*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_7*
is_constant(
І
Agradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_7_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_7*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_7*
is_constant(
ћ
@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_7*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/add_7_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_7_grad/Shape_1*%
_class
loc:@lstm_1/while/add_7*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

%gradients/lstm_1/while/add_7_grad/SumSum;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape7gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_7*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/add_7_grad/ReshapeReshape%gradients/lstm_1/while/add_7_grad/Sum@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_7*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0

'gradients/lstm_1/while/add_7_grad/Sum_1Sum;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape9gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_7*
T0*
	keep_dims( *

Tidx0
р
+gradients/lstm_1/while/add_7_grad/Reshape_1Reshape'gradients/lstm_1/while/add_7_grad/Sum_1)gradients/lstm_1/while/add_7_grad/Shape_1*%
_class
loc:@lstm_1/while/add_7*
T0*
_output_shapes
: *
Tshape0
Њ
'gradients/lstm_1/while/mul_4_grad/ShapeShapelstm_1/while/clip_by_value_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_4
Ї
)gradients/lstm_1/while/mul_4_grad/Shape_1Shapelstm_1/while/Identity_3*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_4
Ќ
=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
І
Agradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_4_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
ћ
@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
Ќ
Cgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_4_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
џ
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_4*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
И
+gradients/lstm_1/while/mul_4_grad/mul/f_accStack*

stack_name *C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Ћ
.gradients/lstm_1/while/mul_4_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_4_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
is_constant(

/gradients/lstm_1/while/mul_4_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_4_grad/mul/RefEnterlstm_1/while/Identity_3^gradients/Add*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
О
7gradients/lstm_1/while/mul_4_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_4_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
is_constant(

.gradients/lstm_1/while/mul_4_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_4_grad/mul/StackPop/RefEnter^gradients/Sub*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_4_grad/mulMul)gradients/lstm_1/while/add_5_grad/Reshape.gradients/lstm_1/while/mul_4_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_4_grad/SumSum%gradients/lstm_1/while/mul_4_grad/mul7gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_4*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/mul_4_grad/ReshapeReshape%gradients/lstm_1/while/mul_4_grad/Sum@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
П
-gradients/lstm_1/while/mul_4_grad/mul_1/f_accStack*

stack_name *H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Д
0gradients/lstm_1/while/mul_4_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_4_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
is_constant(

1gradients/lstm_1/while/mul_4_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_4_grad/mul_1/RefEnterlstm_1/while/clip_by_value_1^gradients/Add*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
Ч
9gradients/lstm_1/while/mul_4_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_4_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
is_constant(

0gradients/lstm_1/while/mul_4_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_4_grad/mul_1/StackPop/RefEnter^gradients/Sub*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
ф
'gradients/lstm_1/while/mul_4_grad/mul_1Mul0gradients/lstm_1/while/mul_4_grad/mul_1/StackPop)gradients/lstm_1/while/add_5_grad/Reshape*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_4_grad/Sum_1Sum'gradients/lstm_1/while/mul_4_grad/mul_19gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_4*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/mul_4_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_4_grad/Sum_1Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ј
'gradients/lstm_1/while/mul_6_grad/ShapeShapelstm_1/while/clip_by_value*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_6
Ё
)gradients/lstm_1/while/mul_6_grad/Shape_1Shapelstm_1/while/Tanh*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_6
Ќ
=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
І
Agradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_6_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
ћ
@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
Ќ
Cgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_6_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
џ
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_6*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
В
+gradients/lstm_1/while/mul_6_grad/mul/f_accStack*

stack_name *=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
Ѕ
.gradients/lstm_1/while/mul_6_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_6_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
is_constant(

/gradients/lstm_1/while/mul_6_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_6_grad/mul/RefEnterlstm_1/while/Tanh^gradients/Add*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
И
7gradients/lstm_1/while/mul_6_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_6_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
is_constant(
ќ
.gradients/lstm_1/while/mul_6_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_6_grad/mul/StackPop/RefEnter^gradients/Sub*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
т
%gradients/lstm_1/while/mul_6_grad/mulMul+gradients/lstm_1/while/add_5_grad/Reshape_1.gradients/lstm_1/while/mul_6_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_6_grad/SumSum%gradients/lstm_1/while/mul_6_grad/mul7gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_6*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/mul_6_grad/ReshapeReshape%gradients/lstm_1/while/mul_6_grad/Sum@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Н
-gradients/lstm_1/while/mul_6_grad/mul_1/f_accStack*

stack_name *F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
В
0gradients/lstm_1/while/mul_6_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_6_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
is_constant(

1gradients/lstm_1/while/mul_6_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_6_grad/mul_1/RefEnterlstm_1/while/clip_by_value^gradients/Add*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
Х
9gradients/lstm_1/while/mul_6_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_6_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
is_constant(

0gradients/lstm_1/while/mul_6_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_6_grad/mul_1/StackPop/RefEnter^gradients/Sub*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
ц
'gradients/lstm_1/while/mul_6_grad/mul_1Mul0gradients/lstm_1/while/mul_6_grad/mul_1/StackPop+gradients/lstm_1/while/add_5_grad/Reshape_1*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_6_grad/Sum_1Sum'gradients/lstm_1/while/mul_6_grad/mul_19gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_6*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/mul_6_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_6_grad/Sum_1Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ё
'gradients/lstm_1/while/mul_8_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_8*
valueB *
_output_shapes
: 
Ђ
)gradients/lstm_1/while/mul_8_grad/Shape_1Shapelstm_1/while/add_6*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_8
Ќ
=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_8*
is_constant(
Ј
Agradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_8_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_8*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_8*
is_constant(
ћ
@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:

7gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_8_grad/Shape@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Г
+gradients/lstm_1/while/mul_8_grad/mul/f_accStack*

stack_name *>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
І
.gradients/lstm_1/while/mul_8_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_8_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
is_constant(

/gradients/lstm_1/while/mul_8_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_8_grad/mul/RefEnterlstm_1/while/add_6^gradients/Add*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
T0*
swap_memory(*
_output_shapes
:
Й
7gradients/lstm_1/while/mul_8_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_8_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
is_constant(
§
.gradients/lstm_1/while/mul_8_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_8_grad/mul/StackPop/RefEnter^gradients/Sub*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_8_grad/mulMul)gradients/lstm_1/while/add_7_grad/Reshape.gradients/lstm_1/while/mul_8_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_8_grad/SumSum%gradients/lstm_1/while/mul_8_grad/mul7gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_8*
T0*
	keep_dims( *

Tidx0
к
)gradients/lstm_1/while/mul_8_grad/ReshapeReshape%gradients/lstm_1/while/mul_8_grad/Sum'gradients/lstm_1/while/mul_8_grad/Shape*%
_class
loc:@lstm_1/while/mul_8*
T0*
_output_shapes
: *
Tshape0
З
-gradients/lstm_1/while/mul_8_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
	elem_type0*
_output_shapes
:
Ќ
0gradients/lstm_1/while/mul_8_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_8_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
is_constant(

1gradients/lstm_1/while/mul_8_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_8_grad/mul_1/RefEnterlstm_1/while/mul_8/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
T0*
swap_memory(*
_output_shapes
:
П
9gradients/lstm_1/while/mul_8_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_8_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
is_constant(
ђ
0gradients/lstm_1/while/mul_8_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_8_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
	elem_type0*
_output_shapes
: 
ф
'gradients/lstm_1/while/mul_8_grad/mul_1Mul0gradients/lstm_1/while/mul_8_grad/mul_1/StackPop)gradients/lstm_1/while/add_7_grad/Reshape*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_8_grad/Sum_1Sum'gradients/lstm_1/while/mul_8_grad/mul_19gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_8*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/mul_8_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_8_grad/Sum_1@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ц
1gradients/lstm_1/while/clip_by_value_1_grad/ShapeShape$lstm_1/while/clip_by_value_1/Minimum*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
З
3gradients/lstm_1/while/clip_by_value_1_grad/Shape_1Const^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB *
_output_shapes
: 
Э
3gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Shape)gradients/lstm_1/while/mul_4_grad/Reshape*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
Н
7gradients/lstm_1/while/clip_by_value_1_grad/zeros/ConstConst^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB
 *    *
_output_shapes
: 

1gradients/lstm_1/while/clip_by_value_1_grad/zerosFill3gradients/lstm_1/while/clip_by_value_1_grad/Shape_27gradients/lstm_1/while/clip_by_value_1_grad/zeros/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ
т
>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_accStack*

stack_name *Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
ш
Agradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
к
Bgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush	StackPushAgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter$lstm_1/while/clip_by_value_1/Minimum^gradients/Add*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Jgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
П
Agradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopStackPopJgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
д
@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1Stack*

stack_name *J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
м
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1RefEnter@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
is_constant(
О
Dgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush_1	StackPushCgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1lstm_1/while/Const_2^gradients/Add*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
T0*
swap_memory(*
_output_shapes
:
я
Lgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1/RefEnterRefEnter@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
is_constant(
Ђ
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1StackPopLgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
: 
Г
8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqualGreaterEqualAgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopCgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ
Р
Ggradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
Я
Jgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
is_constant(
Ю
Kgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPush	StackPushJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnter1gradients/lstm_1/while/clip_by_value_1_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
swap_memory(*
_output_shapes
:
т
Sgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
is_constant(

Jgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopStackPopSgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
Щ
Agradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop3gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
2gradients/lstm_1/while/clip_by_value_1_grad/SelectSelect8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ
и
6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ
З
4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/lstm_1/while/clip_by_value_1_grad/SumSum2gradients/lstm_1/while/clip_by_value_1_grad/SelectAgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
	keep_dims( *

Tidx0
Ќ
3gradients/lstm_1/while/clip_by_value_1_grad/ReshapeReshape/gradients/lstm_1/while/clip_by_value_1_grad/SumJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Є
1gradients/lstm_1/while/clip_by_value_1_grad/Sum_1Sum4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Cgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
	keep_dims( *

Tidx0

5gradients/lstm_1/while/clip_by_value_1_grad/Reshape_1Reshape1gradients/lstm_1/while/clip_by_value_1_grad/Sum_13gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
_output_shapes
: *
Tshape0
Р
/gradients/lstm_1/while/clip_by_value_grad/ShapeShape"lstm_1/while/clip_by_value/Minimum*
out_type0*
T0*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/clip_by_value
Г
1gradients/lstm_1/while/clip_by_value_grad/Shape_1Const^gradients/Sub*
dtype0*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB *
_output_shapes
: 
Щ
1gradients/lstm_1/while/clip_by_value_grad/Shape_2Shape)gradients/lstm_1/while/mul_6_grad/Reshape*
out_type0*
T0*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/clip_by_value
Й
5gradients/lstm_1/while/clip_by_value_grad/zeros/ConstConst^gradients/Sub*
dtype0*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    *
_output_shapes
: 

/gradients/lstm_1/while/clip_by_value_grad/zerosFill1gradients/lstm_1/while/clip_by_value_grad/Shape_25gradients/lstm_1/while/clip_by_value_grad/zeros/Const*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
м
<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_accStack*

stack_name *V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
р
?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnterRefEnter<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
а
@gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush	StackPush?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter"lstm_1/while/clip_by_value/Minimum^gradients/Add*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
ѓ
Hgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop/RefEnterRefEnter<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
З
?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopStackPopHgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
Ю
>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1Stack*

stack_name *F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
д
Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1RefEnter>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
is_constant(
Д
Bgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush_1	StackPushAgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1lstm_1/while/Const^gradients/Add*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
T0*
swap_memory(*
_output_shapes
:
ч
Jgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
is_constant(

Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1StackPopJgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
: 
Ћ
6gradients/lstm_1/while/clip_by_value_grad/GreaterEqualGreaterEqual?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopAgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
М
Egradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_accStack*

stack_name *-
_class#
!loc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
Щ
Hgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
is_constant(
Ц
Igradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPush	StackPushHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnter/gradients/lstm_1/while/clip_by_value_grad/Shape^gradients/Add*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
swap_memory(*
_output_shapes
:
м
Qgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
is_constant(

Hgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopStackPopQgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
С
?gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop1gradients/lstm_1/while/clip_by_value_grad/Shape_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Џ
0gradients/lstm_1/while/clip_by_value_grad/SelectSelect6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
в
4gradients/lstm_1/while/clip_by_value_grad/LogicalNot
LogicalNot6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
Џ
2gradients/lstm_1/while/clip_by_value_grad/Select_1Select4gradients/lstm_1/while/clip_by_value_grad/LogicalNot)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ

-gradients/lstm_1/while/clip_by_value_grad/SumSum0gradients/lstm_1/while/clip_by_value_grad/Select?gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
	keep_dims( *

Tidx0
Є
1gradients/lstm_1/while/clip_by_value_grad/ReshapeReshape-gradients/lstm_1/while/clip_by_value_grad/SumHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0

/gradients/lstm_1/while/clip_by_value_grad/Sum_1Sum2gradients/lstm_1/while/clip_by_value_grad/Select_1Agradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
	keep_dims( *

Tidx0

3gradients/lstm_1/while/clip_by_value_grad/Reshape_1Reshape/gradients/lstm_1/while/clip_by_value_grad/Sum_11gradients/lstm_1/while/clip_by_value_grad/Shape_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
_output_shapes
: *
Tshape0
ъ
)gradients/lstm_1/while/Tanh_grad/TanhGradTanhGrad.gradients/lstm_1/while/mul_6_grad/mul/StackPop+gradients/lstm_1/while/mul_6_grad/Reshape_1*$
_class
loc:@lstm_1/while/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
'gradients/lstm_1/while/add_6_grad/ShapeShapelstm_1/while/strided_slice_3*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_6
Ѕ
)gradients/lstm_1/while/add_6_grad/Shape_1Shapelstm_1/while/MatMul_3*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_6
Ќ
=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
І
Agradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_6_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_6*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
ћ
@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
Ќ
Cgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_6_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_6*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
џ
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/add_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_6*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
%gradients/lstm_1/while/add_6_grad/SumSum+gradients/lstm_1/while/mul_8_grad/Reshape_17gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_6*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/add_6_grad/ReshapeReshape%gradients/lstm_1/while/add_6_grad/Sum@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
§
'gradients/lstm_1/while/add_6_grad/Sum_1Sum+gradients/lstm_1/while/mul_8_grad/Reshape_19gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_6*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/add_6_grad/Reshape_1Reshape'gradients/lstm_1/while/add_6_grad/Sum_1Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ф
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ShapeShapelstm_1/while/add_3*
out_type0*
T0*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
Ч
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB *
_output_shapes
: 
ч
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2Shape3gradients/lstm_1/while/clip_by_value_1_grad/Reshape*
out_type0*
T0*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
Э
?gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB
 *    *
_output_shapes
: 
Њ
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosFill;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2?gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
н
Cgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_accStack*

stack_name *P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
ш
Fgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
Ш
Ggradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush	StackPushFgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_3^gradients/Add*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Ogradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
П
Fgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopStackPopOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
с
Egradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
ю
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1RefEnterEgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
а
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush_1	StackPushHgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_3^gradients/Add*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:

Qgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
Д
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1StackPopQgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
: 
Ч
=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual	LessEqualFgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopHgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
а
Ogradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
ч
Rgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
ю
Sgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnter9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape^gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
њ
[gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
Б
Rgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopStackPop[gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
щ
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
:gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectSelect=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
э
>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ
с
<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
О
7gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumSum:gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectIgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
	keep_dims( *

Tidx0
Ь
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeReshape7gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ф
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1Sum<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Kgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
	keep_dims( *

Tidx0
Ј
=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape_1Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
_output_shapes
: *
Tshape0
Э
4gradients/lstm_1/while/Switch_3_grad_1/NextIterationNextIteration+gradients/lstm_1/while/mul_4_grad/Reshape_1*'
_class
loc:@lstm_1/while/Merge_3*
T0*'
_output_shapes
:џџџџџџџџџ
Р
7gradients/lstm_1/while/clip_by_value/Minimum_grad/ShapeShapelstm_1/while/add_1*
out_type0*
T0*
_output_shapes
:*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
У
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB *
_output_shapes
: 
с
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Shape1gradients/lstm_1/while/clip_by_value_grad/Reshape*
out_type0*
T0*
_output_shapes
:*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
Щ
=gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *    *
_output_shapes
: 
Ђ
7gradients/lstm_1/while/clip_by_value/Minimum_grad/zerosFill9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2=gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/Const*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
й
Agradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_accStack*

stack_name *N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
т
Dgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnterRefEnterAgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Т
Egradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush	StackPushDgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_1^gradients/Add*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
ѕ
Mgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterAgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Й
Dgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopStackPopMgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
н
Cgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
ш
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1RefEnterCgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Ъ
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush_1	StackPushFgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_1^gradients/Add*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Ogradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Ў
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1StackPopOgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
: 
П
;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual	LessEqualDgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopFgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
Mgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
с
Pgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterMgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
ц
Qgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnter7gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape^gradients/Add*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
є
Ygradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterMgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Ћ
Pgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopStackPopYgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
с
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
д
8gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectSelect;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
ч
<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot
LogicalNot;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
з
:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Select<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
Ж
5gradients/lstm_1/while/clip_by_value/Minimum_grad/SumSum8gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectGgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
	keep_dims( *

Tidx0
Ф
9gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeReshape5gradients/lstm_1/while/clip_by_value/Minimum_grad/SumPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
М
7gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_1Sum:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Igradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
	keep_dims( *

Tidx0
 
;gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape_1Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_19gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
_output_shapes
: *
Tshape0
Њ
'gradients/lstm_1/while/add_4_grad/ShapeShapelstm_1/while/strided_slice_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_4
Ѕ
)gradients/lstm_1/while/add_4_grad/Shape_1Shapelstm_1/while/MatMul_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_4
Ќ
=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
І
Agradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_4_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_4*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
ћ
@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
Ќ
Cgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_4_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_4*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
џ
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_4*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
%gradients/lstm_1/while/add_4_grad/SumSum)gradients/lstm_1/while/Tanh_grad/TanhGrad7gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_4*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/add_4_grad/ReshapeReshape%gradients/lstm_1/while/add_4_grad/Sum@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_4*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
ћ
'gradients/lstm_1/while/add_4_grad/Sum_1Sum)gradients/lstm_1/while/Tanh_grad/TanhGrad9gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_4*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/add_4_grad/Reshape_1Reshape'gradients/lstm_1/while/add_4_grad/Sum_1Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_4*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Р
1gradients/lstm_1/while/strided_slice_3_grad/ShapeShapelstm_1/while/TensorArrayReadV3*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/strided_slice_3
Л
Bgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_3*
	elem_type0*
_output_shapes
:
Х
Egradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
is_constant(
Ф
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_3_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*
swap_memory(*
_output_shapes
:
и
Ngradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
is_constant(

Egradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
	elem_type0*
_output_shapes
:
ц
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
	elem_type0*
_output_shapes
:
ђ
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
is_constant(
т
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_3/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
is_constant(
М
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_3/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_3/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
	elem_type0*
_output_shapes
:
ё
<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_6_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_3

1gradients/lstm_1/while/MatMul_3_grad/MatMul/EnterEnterlstm_1/strided_slice_7*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_3*
is_constant(

+gradients/lstm_1/while/MatMul_3_grad/MatMulMatMul+gradients/lstm_1/while/add_6_grad/Reshape_11gradients/lstm_1/while/MatMul_3_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_3*
T0*'
_output_shapes
:џџџџџџџџџ
О
3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
Й
6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
is_constant(

7gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnterlstm_1/while/mul_7^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
Ь
?gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
is_constant(

6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
	elem_type0*'
_output_shapes
:џџџџџџџџџ

-gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_6_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_3*
T0*
_output_shapes

:
 
'gradients/lstm_1/while/add_3_grad/ShapeShapelstm_1/while/mul_3*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_3
Ѓ
)gradients/lstm_1/while/add_3_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_3*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_3*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_3*
is_constant(
І
Agradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_3_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_3*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_3*
is_constant(
ћ
@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_3*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_3_grad/Shape_1*%
_class
loc:@lstm_1/while/add_3*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

%gradients/lstm_1/while/add_3_grad/SumSum;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape7gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_3*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/add_3_grad/ReshapeReshape%gradients/lstm_1/while/add_3_grad/Sum@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_3*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0

'gradients/lstm_1/while/add_3_grad/Sum_1Sum;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape9gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_3*
T0*
	keep_dims( *

Tidx0
р
+gradients/lstm_1/while/add_3_grad/Reshape_1Reshape'gradients/lstm_1/while/add_3_grad/Sum_1)gradients/lstm_1/while/add_3_grad/Shape_1*%
_class
loc:@lstm_1/while/add_3*
T0*
_output_shapes
: *
Tshape0
 
'gradients/lstm_1/while/add_1_grad/ShapeShapelstm_1/while/mul_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_1
Ѓ
)gradients/lstm_1/while/add_1_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_1*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_1*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_1*
is_constant(
І
Agradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_1_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_1*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_1*
is_constant(
ћ
@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_1_grad/Shape_1*%
_class
loc:@lstm_1/while/add_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

%gradients/lstm_1/while/add_1_grad/SumSum9gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape7gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_1*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/add_1_grad/ReshapeReshape%gradients/lstm_1/while/add_1_grad/Sum@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_1*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0

'gradients/lstm_1/while/add_1_grad/Sum_1Sum9gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape9gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_1*
T0*
	keep_dims( *

Tidx0
р
+gradients/lstm_1/while/add_1_grad/Reshape_1Reshape'gradients/lstm_1/while/add_1_grad/Sum_1)gradients/lstm_1/while/add_1_grad/Shape_1*%
_class
loc:@lstm_1/while/add_1*
T0*
_output_shapes
: *
Tshape0
Р
1gradients/lstm_1/while/strided_slice_2_grad/ShapeShapelstm_1/while/TensorArrayReadV3*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/strided_slice_2
Л
Bgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_2*
	elem_type0*
_output_shapes
:
Х
Egradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
is_constant(
Ф
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_2_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
T0*
swap_memory(*
_output_shapes
:
и
Ngradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
is_constant(

Egradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
	elem_type0*
_output_shapes
:
ц
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
	elem_type0*
_output_shapes
:
ђ
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
is_constant(
т
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_2/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
is_constant(
М
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_2/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_2/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
	elem_type0*
_output_shapes
:
ё
<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_4_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_2

1gradients/lstm_1/while/MatMul_2_grad/MatMul/EnterEnterlstm_1/strided_slice_6*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_2*
is_constant(

+gradients/lstm_1/while/MatMul_2_grad/MatMulMatMul+gradients/lstm_1/while/add_4_grad/Reshape_11gradients/lstm_1/while/MatMul_2_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_2*
T0*'
_output_shapes
:џџџџџџџџџ
О
3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
Й
6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
is_constant(

7gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnterlstm_1/while/mul_5^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
T0*
swap_memory(*
_output_shapes
:
Ь
?gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
is_constant(

6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
	elem_type0*'
_output_shapes
:џџџџџџџџџ

-gradients/lstm_1/while/MatMul_2_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_4_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_2*
T0*
_output_shapes

:
Ѕ
'gradients/lstm_1/while/mul_7_grad/ShapeShapelstm_1/while/Identity_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_7
Ѓ
)gradients/lstm_1/while/mul_7_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_7*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_7*
is_constant(
І
Agradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_7_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_7*
is_constant(
ћ
@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_7_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_7*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Е
+gradients/lstm_1/while/mul_7_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
	elem_type0*
_output_shapes
:
Ј
.gradients/lstm_1/while/mul_7_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_7_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
is_constant(

/gradients/lstm_1/while/mul_7_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_7_grad/mul/RefEnterlstm_1/while/mul_7/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
T0*
swap_memory(*
_output_shapes
:
Л
7gradients/lstm_1/while/mul_7_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_7_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
is_constant(
ю
.gradients/lstm_1/while/mul_7_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_7_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
	elem_type0*
_output_shapes
: 
т
%gradients/lstm_1/while/mul_7_grad/mulMul+gradients/lstm_1/while/MatMul_3_grad/MatMul.gradients/lstm_1/while/mul_7_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_7_grad/SumSum%gradients/lstm_1/while/mul_7_grad/mul7gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_7*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/mul_7_grad/ReshapeReshape%gradients/lstm_1/while/mul_7_grad/Sum@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
К
-gradients/lstm_1/while/mul_7_grad/mul_1/f_accStack*

stack_name *C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
Џ
0gradients/lstm_1/while/mul_7_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_7_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
is_constant(

1gradients/lstm_1/while/mul_7_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_7_grad/mul_1/RefEnterlstm_1/while/Identity_2^gradients/Add*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
Т
9gradients/lstm_1/while/mul_7_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_7_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
is_constant(

0gradients/lstm_1/while/mul_7_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_7_grad/mul_1/StackPop/RefEnter^gradients/Sub*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
ц
'gradients/lstm_1/while/mul_7_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_3_grad/MatMul*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_7_grad/Sum_1Sum'gradients/lstm_1/while/mul_7_grad/mul_19gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_7*
T0*
	keep_dims( *

Tidx0
р
+gradients/lstm_1/while/mul_7_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_7_grad/Sum_1)gradients/lstm_1/while/mul_7_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_7*
T0*
_output_shapes
: *
Tshape0
Е
0gradients/lstm_1/while/MatMul_3/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
valueB*    *
_output_shapes

:
Њ
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
is_constant( 

2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_3/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0* 
_output_shapes
:: *
N
ч
1gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*(
_output_shapes
::
ђ
.gradients/lstm_1/while/MatMul_3/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_3_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
в
8gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_3/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
Ц
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
Ё
'gradients/lstm_1/while/mul_3_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_3*
valueB *
_output_shapes
: 
Ђ
)gradients/lstm_1/while/mul_3_grad/Shape_1Shapelstm_1/while/add_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_3
Ќ
=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_3*
is_constant(
Ј
Agradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_3_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_3*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_3*
is_constant(
ћ
@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:

7gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_3_grad/Shape@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Г
+gradients/lstm_1/while/mul_3_grad/mul/f_accStack*

stack_name *>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
І
.gradients/lstm_1/while/mul_3_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_3_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
is_constant(

/gradients/lstm_1/while/mul_3_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_3_grad/mul/RefEnterlstm_1/while/add_2^gradients/Add*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
T0*
swap_memory(*
_output_shapes
:
Й
7gradients/lstm_1/while/mul_3_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_3_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
is_constant(
§
.gradients/lstm_1/while/mul_3_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_3_grad/mul/StackPop/RefEnter^gradients/Sub*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_3_grad/mulMul)gradients/lstm_1/while/add_3_grad/Reshape.gradients/lstm_1/while/mul_3_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_3_grad/SumSum%gradients/lstm_1/while/mul_3_grad/mul7gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_3*
T0*
	keep_dims( *

Tidx0
к
)gradients/lstm_1/while/mul_3_grad/ReshapeReshape%gradients/lstm_1/while/mul_3_grad/Sum'gradients/lstm_1/while/mul_3_grad/Shape*%
_class
loc:@lstm_1/while/mul_3*
T0*
_output_shapes
: *
Tshape0
З
-gradients/lstm_1/while/mul_3_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
	elem_type0*
_output_shapes
:
Ќ
0gradients/lstm_1/while/mul_3_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_3_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
is_constant(

1gradients/lstm_1/while/mul_3_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_3_grad/mul_1/RefEnterlstm_1/while/mul_3/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
T0*
swap_memory(*
_output_shapes
:
П
9gradients/lstm_1/while/mul_3_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_3_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
is_constant(
ђ
0gradients/lstm_1/while/mul_3_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_3_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
	elem_type0*
_output_shapes
: 
ф
'gradients/lstm_1/while/mul_3_grad/mul_1Mul0gradients/lstm_1/while/mul_3_grad/mul_1/StackPop)gradients/lstm_1/while/add_3_grad/Reshape*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_3_grad/Sum_1Sum'gradients/lstm_1/while/mul_3_grad/mul_19gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_3*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/mul_3_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_3_grad/Sum_1@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ё
'gradients/lstm_1/while/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_1*
valueB *
_output_shapes
: 
 
)gradients/lstm_1/while/mul_1_grad/Shape_1Shapelstm_1/while/add*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_1
Ќ
=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_1*
is_constant(
Ј
Agradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_1_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_1*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_1*
is_constant(
ћ
@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:

7gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_1_grad/Shape@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Б
+gradients/lstm_1/while/mul_1_grad/mul/f_accStack*

stack_name *<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
Є
.gradients/lstm_1/while/mul_1_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_1_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
is_constant(

/gradients/lstm_1/while/mul_1_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_1_grad/mul/RefEnterlstm_1/while/add^gradients/Add*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
T0*
swap_memory(*
_output_shapes
:
З
7gradients/lstm_1/while/mul_1_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_1_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
is_constant(
ћ
.gradients/lstm_1/while/mul_1_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_1_grad/mulMul)gradients/lstm_1/while/add_1_grad/Reshape.gradients/lstm_1/while/mul_1_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_1_grad/SumSum%gradients/lstm_1/while/mul_1_grad/mul7gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_1*
T0*
	keep_dims( *

Tidx0
к
)gradients/lstm_1/while/mul_1_grad/ReshapeReshape%gradients/lstm_1/while/mul_1_grad/Sum'gradients/lstm_1/while/mul_1_grad/Shape*%
_class
loc:@lstm_1/while/mul_1*
T0*
_output_shapes
: *
Tshape0
З
-gradients/lstm_1/while/mul_1_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
	elem_type0*
_output_shapes
:
Ќ
0gradients/lstm_1/while/mul_1_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_1_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
is_constant(

1gradients/lstm_1/while/mul_1_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_1_grad/mul_1/RefEnterlstm_1/while/mul_1/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
T0*
swap_memory(*
_output_shapes
:
П
9gradients/lstm_1/while/mul_1_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_1_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
is_constant(
ђ
0gradients/lstm_1/while/mul_1_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
	elem_type0*
_output_shapes
: 
ф
'gradients/lstm_1/while/mul_1_grad/mul_1Mul0gradients/lstm_1/while/mul_1_grad/mul_1/StackPop)gradients/lstm_1/while/add_1_grad/Reshape*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_1_grad/Sum_1Sum'gradients/lstm_1/while/mul_1_grad/mul_19gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_1*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/mul_1_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_1_grad/Sum_1@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ѕ
'gradients/lstm_1/while/mul_5_grad/ShapeShapelstm_1/while/Identity_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_5
Ѓ
)gradients/lstm_1/while/mul_5_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_5*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_5*
is_constant(
І
Agradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_5_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_5*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_5*
is_constant(
ћ
@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_5_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_5*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Е
+gradients/lstm_1/while/mul_5_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
	elem_type0*
_output_shapes
:
Ј
.gradients/lstm_1/while/mul_5_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_5_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
is_constant(

/gradients/lstm_1/while/mul_5_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_5_grad/mul/RefEnterlstm_1/while/mul_5/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
T0*
swap_memory(*
_output_shapes
:
Л
7gradients/lstm_1/while/mul_5_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_5_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
is_constant(
ю
.gradients/lstm_1/while/mul_5_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_5_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
	elem_type0*
_output_shapes
: 
т
%gradients/lstm_1/while/mul_5_grad/mulMul+gradients/lstm_1/while/MatMul_2_grad/MatMul.gradients/lstm_1/while/mul_5_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_5_grad/SumSum%gradients/lstm_1/while/mul_5_grad/mul7gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_5*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/mul_5_grad/ReshapeReshape%gradients/lstm_1/while/mul_5_grad/Sum@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
ц
'gradients/lstm_1/while/mul_5_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_2_grad/MatMul*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_5_grad/Sum_1Sum'gradients/lstm_1/while/mul_5_grad/mul_19gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_5*
T0*
	keep_dims( *

Tidx0
р
+gradients/lstm_1/while/mul_5_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_5_grad/Sum_1)gradients/lstm_1/while/mul_5_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_5*
T0*
_output_shapes
: *
Tshape0
Е
0gradients/lstm_1/while/MatMul_2/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
valueB*    *
_output_shapes

:
Њ
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
is_constant( 

2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_2/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0* 
_output_shapes
:: *
N
ч
1gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*(
_output_shapes
::
ђ
.gradients/lstm_1/while/MatMul_2/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_2_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
в
8gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_2/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
Ц
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
Ї
+gradients/lstm_1/strided_slice_7_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_7*
valueB"      *
_output_shapes
:
Ю
6gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_22gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_7
Њ
'gradients/lstm_1/while/add_2_grad/ShapeShapelstm_1/while/strided_slice_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_2
Ѕ
)gradients/lstm_1/while/add_2_grad/Shape_1Shapelstm_1/while/MatMul_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_2
Ќ
=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
І
Agradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_2_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_2*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
ћ
@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
Ќ
Cgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_2_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_2*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
џ
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_2*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
%gradients/lstm_1/while/add_2_grad/SumSum+gradients/lstm_1/while/mul_3_grad/Reshape_17gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_2*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/add_2_grad/ReshapeReshape%gradients/lstm_1/while/add_2_grad/Sum@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_2*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
§
'gradients/lstm_1/while/add_2_grad/Sum_1Sum+gradients/lstm_1/while/mul_3_grad/Reshape_19gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_2*
T0*
	keep_dims( *

Tidx0

+gradients/lstm_1/while/add_2_grad/Reshape_1Reshape'gradients/lstm_1/while/add_2_grad/Sum_1Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_2*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Є
%gradients/lstm_1/while/add_grad/ShapeShapelstm_1/while/strided_slice*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/while/add

'gradients/lstm_1/while/add_grad/Shape_1Shapelstm_1/while/MatMul*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/while/add
Ј
;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_accStack*

stack_name *#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
Ћ
>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnterRefEnter;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(

?gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush	StackPush>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter%gradients/lstm_1/while/add_grad/Shape^gradients/Add*#
_class
loc:@lstm_1/while/add*
T0*
swap_memory(*
_output_shapes
:
О
Ggradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
ѕ
>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPopStackPopGgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
Њ
=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
Џ
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1RefEnter=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
Є
Agradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush_1	StackPush@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1'gradients/lstm_1/while/add_grad/Shape_1^gradients/Add*#
_class
loc:@lstm_1/while/add*
T0*
swap_memory(*
_output_shapes
:
Т
Igradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
љ
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1StackPopIgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
В
5gradients/lstm_1/while/add_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1*#
_class
loc:@lstm_1/while/add*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѓ
#gradients/lstm_1/while/add_grad/SumSum+gradients/lstm_1/while/mul_1_grad/Reshape_15gradients/lstm_1/while/add_grad/BroadcastGradientArgs*
_output_shapes
:*#
_class
loc:@lstm_1/while/add*
T0*
	keep_dims( *

Tidx0
ќ
'gradients/lstm_1/while/add_grad/ReshapeReshape#gradients/lstm_1/while/add_grad/Sum>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop*#
_class
loc:@lstm_1/while/add*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
ї
%gradients/lstm_1/while/add_grad/Sum_1Sum+gradients/lstm_1/while/mul_1_grad/Reshape_17gradients/lstm_1/while/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*#
_class
loc:@lstm_1/while/add*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/add_grad/Reshape_1Reshape%gradients/lstm_1/while/add_grad/Sum_1@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1*#
_class
loc:@lstm_1/while/add*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Ї
+gradients/lstm_1/strided_slice_6_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_6*
valueB"      *
_output_shapes
:
Ю
6gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_22gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_6
Р
1gradients/lstm_1/while/strided_slice_1_grad/ShapeShapelstm_1/while/TensorArrayReadV3*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/strided_slice_1
Л
Bgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_1*
	elem_type0*
_output_shapes
:
Х
Egradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
is_constant(
Ф
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_1_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
T0*
swap_memory(*
_output_shapes
:
и
Ngradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
is_constant(

Egradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
	elem_type0*
_output_shapes
:
ц
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
	elem_type0*
_output_shapes
:
ђ
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
is_constant(
т
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_1/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
is_constant(
М
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_1/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_1/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
	elem_type0*
_output_shapes
:
ё
<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_2_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_1

1gradients/lstm_1/while/MatMul_1_grad/MatMul/EnterEnterlstm_1/strided_slice_5*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_1*
is_constant(

+gradients/lstm_1/while/MatMul_1_grad/MatMulMatMul+gradients/lstm_1/while/add_2_grad/Reshape_11gradients/lstm_1/while/MatMul_1_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ
О
3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
Й
6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
is_constant(

7gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnterlstm_1/while/mul_2^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
T0*
swap_memory(*
_output_shapes
:
Ь
?gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
is_constant(

6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
	elem_type0*'
_output_shapes
:џџџџџџџџџ

-gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_2_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_1*
T0*
_output_shapes

:
М
/gradients/lstm_1/while/strided_slice_grad/ShapeShapelstm_1/while/TensorArrayReadV3*
out_type0*
T0*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/strided_slice
З
@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_accStack*

stack_name *-
_class#
!loc:@lstm_1/while/strided_slice*
	elem_type0*
_output_shapes
:
П
Cgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnterRefEnter@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/strided_slice*
is_constant(
М
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush	StackPushCgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter/gradients/lstm_1/while/strided_slice_grad/Shape^gradients/Add*-
_class#
!loc:@lstm_1/while/strided_slice*
T0*
swap_memory(*
_output_shapes
:
в
Lgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop/RefEnterRefEnter@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/strided_slice*
is_constant(

Cgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopStackPopLgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*-
_class#
!loc:@lstm_1/while/strided_slice*
	elem_type0*
_output_shapes
:
р
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
	elem_type0*
_output_shapes
:
ъ
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
is_constant(
и
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_1	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1 lstm_1/while/strided_slice/stack^gradients/Add*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
T0*
swap_memory(*
_output_shapes
:
§
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
is_constant(
Д
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
	elem_type0*
_output_shapes
:
т
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
	elem_type0*
_output_shapes
:
ь
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
is_constant(
м
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_2	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2"lstm_1/while/strided_slice/stack_1^gradients/Add*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
T0*
swap_memory(*
_output_shapes
:
џ
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
is_constant(
Ж
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
	elem_type0*
_output_shapes
:
т
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
	elem_type0*
_output_shapes
:
ь
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
is_constant(
м
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_3	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3"lstm_1/while/strided_slice/stack_2^gradients/Add*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
T0*
swap_memory(*
_output_shapes
:
џ
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
is_constant(
Ж
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
	elem_type0*
_output_shapes
:
у
:gradients/lstm_1/while/strided_slice_grad/StridedSliceGradStridedSliceGradCgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3'gradients/lstm_1/while/add_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *-
_class#
!loc:@lstm_1/while/strided_slice

/gradients/lstm_1/while/MatMul_grad/MatMul/EnterEnterlstm_1/strided_slice_4*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/MatMul*
is_constant(

)gradients/lstm_1/while/MatMul_grad/MatMulMatMul)gradients/lstm_1/while/add_grad/Reshape_1/gradients/lstm_1/while/MatMul_grad/MatMul/Enter*
transpose_b(*
transpose_a( *&
_class
loc:@lstm_1/while/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
И
1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_accStack*

stack_name *=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
Б
4gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnterRefEnter1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
is_constant(

5gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPush	StackPush4gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnterlstm_1/while/mul^gradients/Add*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
T0*
swap_memory(*
_output_shapes
:
Ф
=gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnter1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
is_constant(

4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPopStackPop=gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
	elem_type0*'
_output_shapes
:џџџџџџџџџ

+gradients/lstm_1/while/MatMul_grad/MatMul_1MatMul4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop)gradients/lstm_1/while/add_grad/Reshape_1*
transpose_b( *
transpose_a(*&
_class
loc:@lstm_1/while/MatMul*
T0*
_output_shapes

:
Ѕ
'gradients/lstm_1/while/mul_2_grad/ShapeShapelstm_1/while/Identity_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_2
Ѓ
)gradients/lstm_1/while/mul_2_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_2*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_2*
is_constant(
І
Agradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_2_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_2*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_2*
is_constant(
ћ
@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_2_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_2*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Е
+gradients/lstm_1/while/mul_2_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
	elem_type0*
_output_shapes
:
Ј
.gradients/lstm_1/while/mul_2_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_2_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
is_constant(

/gradients/lstm_1/while/mul_2_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_2_grad/mul/RefEnterlstm_1/while/mul_2/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
T0*
swap_memory(*
_output_shapes
:
Л
7gradients/lstm_1/while/mul_2_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_2_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
is_constant(
ю
.gradients/lstm_1/while/mul_2_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
	elem_type0*
_output_shapes
: 
т
%gradients/lstm_1/while/mul_2_grad/mulMul+gradients/lstm_1/while/MatMul_1_grad/MatMul.gradients/lstm_1/while/mul_2_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_2_grad/SumSum%gradients/lstm_1/while/mul_2_grad/mul7gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_2*
T0*
	keep_dims( *

Tidx0

)gradients/lstm_1/while/mul_2_grad/ReshapeReshape%gradients/lstm_1/while/mul_2_grad/Sum@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
ц
'gradients/lstm_1/while/mul_2_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_1_grad/MatMul*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_2_grad/Sum_1Sum'gradients/lstm_1/while/mul_2_grad/mul_19gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_2*
T0*
	keep_dims( *

Tidx0
р
+gradients/lstm_1/while/mul_2_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_2_grad/Sum_1)gradients/lstm_1/while/mul_2_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_2*
T0*
_output_shapes
: *
Tshape0
Е
0gradients/lstm_1/while/MatMul_1/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
valueB*    *
_output_shapes

:
Њ
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
is_constant( 

2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_1/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0* 
_output_shapes
:: *
N
ч
1gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*(
_output_shapes
::
ђ
.gradients/lstm_1/while/MatMul_1/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_1_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
в
8gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_1/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
Ц
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
њ
gradients/AddN_3AddN<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad:gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*'
_output_shapes
:џџџџџџџџџ*
N
І
Ugradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
б
Wgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(

Ogradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Ugradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterWgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
source	gradients*'
_class
loc:@lstm_1/TensorArray_1*
_output_shapes

::
Ь
Kgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityWgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1P^gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
Н
Qgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ogradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopgradients/AddN_3Kgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
Ё
%gradients/lstm_1/while/mul_grad/ShapeShapelstm_1/while/Identity_2*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/while/mul

'gradients/lstm_1/while/mul_grad/Shape_1Const^gradients/Sub*
dtype0*#
_class
loc:@lstm_1/while/mul*
valueB *
_output_shapes
: 
Ј
;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_accStack*

stack_name *#
_class
loc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
Ћ
>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnterRefEnter;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/mul*
is_constant(

?gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPush	StackPush>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnter%gradients/lstm_1/while/mul_grad/Shape^gradients/Add*#
_class
loc:@lstm_1/while/mul*
T0*
swap_memory(*
_output_shapes
:
О
Ggradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/mul*
is_constant(
ѕ
>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopStackPopGgradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:

5gradients/lstm_1/while/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop'gradients/lstm_1/while/mul_grad/Shape_1*#
_class
loc:@lstm_1/while/mul*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Џ
)gradients/lstm_1/while/mul_grad/mul/f_accStack*

stack_name *<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
	elem_type0*
_output_shapes
:
 
,gradients/lstm_1/while/mul_grad/mul/RefEnterRefEnter)gradients/lstm_1/while/mul_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
is_constant(

-gradients/lstm_1/while/mul_grad/mul/StackPush	StackPush,gradients/lstm_1/while/mul_grad/mul/RefEnterlstm_1/while/mul/y^gradients/Add*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
T0*
swap_memory(*
_output_shapes
:
Г
5gradients/lstm_1/while/mul_grad/mul/StackPop/RefEnterRefEnter)gradients/lstm_1/while/mul_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
is_constant(
ц
,gradients/lstm_1/while/mul_grad/mul/StackPopStackPop5gradients/lstm_1/while/mul_grad/mul/StackPop/RefEnter^gradients/Sub*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
	elem_type0*
_output_shapes
: 
к
#gradients/lstm_1/while/mul_grad/mulMul)gradients/lstm_1/while/MatMul_grad/MatMul,gradients/lstm_1/while/mul_grad/mul/StackPop*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:џџџџџџџџџ
ы
#gradients/lstm_1/while/mul_grad/SumSum#gradients/lstm_1/while/mul_grad/mul5gradients/lstm_1/while/mul_grad/BroadcastGradientArgs*
_output_shapes
:*#
_class
loc:@lstm_1/while/mul*
T0*
	keep_dims( *

Tidx0
ќ
'gradients/lstm_1/while/mul_grad/ReshapeReshape#gradients/lstm_1/while/mul_grad/Sum>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
р
%gradients/lstm_1/while/mul_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop)gradients/lstm_1/while/MatMul_grad/MatMul*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:џџџџџџџџџ
ё
%gradients/lstm_1/while/mul_grad/Sum_1Sum%gradients/lstm_1/while/mul_grad/mul_17gradients/lstm_1/while/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*#
_class
loc:@lstm_1/while/mul*
T0*
	keep_dims( *

Tidx0
и
)gradients/lstm_1/while/mul_grad/Reshape_1Reshape%gradients/lstm_1/while/mul_grad/Sum_1'gradients/lstm_1/while/mul_grad/Shape_1*#
_class
loc:@lstm_1/while/mul*
T0*
_output_shapes
: *
Tshape0
Б
.gradients/lstm_1/while/MatMul/Enter_grad/b_accConst*
dtype0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
valueB*    *
_output_shapes

:
Є
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter.gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
is_constant( 

0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2Merge0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_16gradients/lstm_1/while/MatMul/Enter_grad/NextIteration*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0* 
_output_shapes
:: *
N
с
/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*(
_output_shapes
::
ъ
,gradients/lstm_1/while/MatMul/Enter_grad/AddAdd1gradients/lstm_1/while/MatMul/Enter_grad/Switch:1+gradients/lstm_1/while/MatMul_grad/MatMul_1*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
Ь
6gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration,gradients/lstm_1/while/MatMul/Enter_grad/Add*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
Р
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit/gradients/lstm_1/while/MatMul/Enter_grad/Switch*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
Ї
+gradients/lstm_1/strided_slice_5_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_5*
valueB"      *
_output_shapes
:
Ю
6gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_22gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_5
Љ
;gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*
valueB
 *    *
_output_shapes
: 
Б
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter;gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant( 

=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Cgradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIteration*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: : *
N
ц
<gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: : 

9gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/AddAdd>gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch:1Qgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
й
Cgradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration9gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Add*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
Э
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit<gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
Є
gradients/AddN_4AddN)gradients/lstm_1/while/mul_7_grad/Reshape)gradients/lstm_1/while/mul_5_grad/Reshape)gradients/lstm_1/while/mul_2_grad/Reshape'gradients/lstm_1/while/mul_grad/Reshape*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ*
N
Ї
+gradients/lstm_1/strided_slice_4_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_4*
valueB"      *
_output_shapes
:
Ь
6gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_20gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_4
М
rgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArray_1=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*'
_class
loc:@lstm_1/TensorArray_1*
_output_shapes

::
ј
ngradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3s^gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
њ
dgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3rgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lstm_1/TensorArrayUnstack/rangengradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*+
_output_shapes
:(џџџџџџџџџ*
element_shape:
В
4gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_4*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:џџџџџџџџџ
е
gradients/AddN_5AddN6gradients/lstm_1/strided_slice_7_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_6_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_5_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*)
_class
loc:@lstm_1/strided_slice_7*
T0*
_output_shapes

:*
N
Ї
1gradients/lstm_1/transpose_grad/InvertPermutationInvertPermutationlstm_1/transpose/perm*#
_class
loc:@lstm_1/transpose*
T0*
_output_shapes
:
З
)gradients/lstm_1/transpose_grad/transpose	Transposedgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV31gradients/lstm_1/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:џџџџџџџџџ(*#
_class
loc:@lstm_1/transpose

!gradients/lstm_1/concat_grad/RankConst*
dtype0* 
_class
loc:@lstm_1/concat*
value	B :*
_output_shapes
: 
І
 gradients/lstm_1/concat_grad/modFloorModlstm_1/concat/axis!gradients/lstm_1/concat_grad/Rank* 
_class
loc:@lstm_1/concat*
T0*
_output_shapes
: 

"gradients/lstm_1/concat_grad/ShapeShapelstm_1/Reshape_1*
out_type0*
T0*
_output_shapes
:* 
_class
loc:@lstm_1/concat
ч
#gradients/lstm_1/concat_grad/ShapeNShapeNlstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7*
out_type0* 
_class
loc:@lstm_1/concat*
T0*,
_output_shapes
::::*
N
Ю
)gradients/lstm_1/concat_grad/ConcatOffsetConcatOffset gradients/lstm_1/concat_grad/mod#gradients/lstm_1/concat_grad/ShapeN%gradients/lstm_1/concat_grad/ShapeN:1%gradients/lstm_1/concat_grad/ShapeN:2%gradients/lstm_1/concat_grad/ShapeN:3* 
_class
loc:@lstm_1/concat*,
_output_shapes
::::*
N

"gradients/lstm_1/concat_grad/SliceSlice)gradients/lstm_1/transpose_grad/transpose)gradients/lstm_1/concat_grad/ConcatOffset#gradients/lstm_1/concat_grad/ShapeN*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:џџџџџџџџџ(

$gradients/lstm_1/concat_grad/Slice_1Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:1%gradients/lstm_1/concat_grad/ShapeN:1*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:џџџџџџџџџ(

$gradients/lstm_1/concat_grad/Slice_2Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:2%gradients/lstm_1/concat_grad/ShapeN:2*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:џџџџџџџџџ(

$gradients/lstm_1/concat_grad/Slice_3Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:3%gradients/lstm_1/concat_grad/ShapeN:3*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:џџџџџџџџџ(

%gradients/lstm_1/Reshape_1_grad/ShapeShapelstm_1/BiasAdd*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/Reshape_1
т
'gradients/lstm_1/Reshape_1_grad/ReshapeReshape"gradients/lstm_1/concat_grad/Slice%gradients/lstm_1/Reshape_1_grad/Shape*#
_class
loc:@lstm_1/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0

%gradients/lstm_1/Reshape_3_grad/ShapeShapelstm_1/BiasAdd_1*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/Reshape_3
ф
'gradients/lstm_1/Reshape_3_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_1%gradients/lstm_1/Reshape_3_grad/Shape*#
_class
loc:@lstm_1/Reshape_3*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0

%gradients/lstm_1/Reshape_5_grad/ShapeShapelstm_1/BiasAdd_2*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/Reshape_5
ф
'gradients/lstm_1/Reshape_5_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_2%gradients/lstm_1/Reshape_5_grad/Shape*#
_class
loc:@lstm_1/Reshape_5*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0

%gradients/lstm_1/Reshape_7_grad/ShapeShapelstm_1/BiasAdd_3*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/Reshape_7
ф
'gradients/lstm_1/Reshape_7_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_3%gradients/lstm_1/Reshape_7_grad/Shape*#
_class
loc:@lstm_1/Reshape_7*
T0*'
_output_shapes
:џџџџџџџџџ*
Tshape0
Р
)gradients/lstm_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_1_grad/Reshape*!
_class
loc:@lstm_1/BiasAdd*
T0*
_output_shapes
:*
data_formatNHWC
Ф
+gradients/lstm_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_3_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_1*
T0*
_output_shapes
:*
data_formatNHWC
Ф
+gradients/lstm_1/BiasAdd_2_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_5_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_2*
T0*
_output_shapes
:*
data_formatNHWC
Ф
+gradients/lstm_1/BiasAdd_3_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_7_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_3*
T0*
_output_shapes
:*
data_formatNHWC
ц
#gradients/lstm_1/MatMul_grad/MatMulMatMul'gradients/lstm_1/Reshape_1_grad/Reshapelstm_1/strided_slice*
transpose_b(*
transpose_a( * 
_class
loc:@lstm_1/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
й
%gradients/lstm_1/MatMul_grad/MatMul_1MatMullstm_1/Reshape'gradients/lstm_1/Reshape_1_grad/Reshape*
transpose_b( *
transpose_a(* 
_class
loc:@lstm_1/MatMul*
T0*
_output_shapes

:
 
+gradients/lstm_1/strided_slice_8_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_8*
valueB:*
_output_shapes
:
С
6gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2)gradients/lstm_1/BiasAdd_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_8
ь
%gradients/lstm_1/MatMul_1_grad/MatMulMatMul'gradients/lstm_1/Reshape_3_grad/Reshapelstm_1/strided_slice_1*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ
п
'gradients/lstm_1/MatMul_1_grad/MatMul_1MatMullstm_1/Reshape_2'gradients/lstm_1/Reshape_3_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_1*
T0*
_output_shapes

:
 
+gradients/lstm_1/strided_slice_9_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_9*
valueB:*
_output_shapes
:
У
6gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2+gradients/lstm_1/BiasAdd_1_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_9
ь
%gradients/lstm_1/MatMul_2_grad/MatMulMatMul'gradients/lstm_1/Reshape_5_grad/Reshapelstm_1/strided_slice_2*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_2*
T0*'
_output_shapes
:џџџџџџџџџ
п
'gradients/lstm_1/MatMul_2_grad/MatMul_1MatMullstm_1/Reshape_4'gradients/lstm_1/Reshape_5_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_2*
T0*
_output_shapes

:
Ђ
,gradients/lstm_1/strided_slice_10_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_10*
valueB:*
_output_shapes
:
Щ
7gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2+gradients/lstm_1/BiasAdd_2_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask **
_class 
loc:@lstm_1/strided_slice_10
ь
%gradients/lstm_1/MatMul_3_grad/MatMulMatMul'gradients/lstm_1/Reshape_7_grad/Reshapelstm_1/strided_slice_3*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_3*
T0*'
_output_shapes
:џџџџџџџџџ
п
'gradients/lstm_1/MatMul_3_grad/MatMul_1MatMullstm_1/Reshape_6'gradients/lstm_1/Reshape_7_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_3*
T0*
_output_shapes

:
Ђ
,gradients/lstm_1/strided_slice_11_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_11*
valueB:*
_output_shapes
:
Щ
7gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2+gradients/lstm_1/BiasAdd_3_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask **
_class 
loc:@lstm_1/strided_slice_11
Ѓ
)gradients/lstm_1/strided_slice_grad/ShapeConst*
dtype0*'
_class
loc:@lstm_1/strided_slice*
valueB"      *
_output_shapes
:
Е
4gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad)gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2%gradients/lstm_1/MatMul_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *'
_class
loc:@lstm_1/strided_slice
Ї
+gradients/lstm_1/strided_slice_1_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_1*
valueB"      *
_output_shapes
:
У
6gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2'gradients/lstm_1/MatMul_1_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_1
Ї
+gradients/lstm_1/strided_slice_2_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_2*
valueB"      *
_output_shapes
:
У
6gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2'gradients/lstm_1/MatMul_2_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_2
Ї
+gradients/lstm_1/strided_slice_3_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_3*
valueB"      *
_output_shapes
:
У
6gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2'gradients/lstm_1/MatMul_3_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_3
г
gradients/AddN_6AddN6gradients/lstm_1/strided_slice_8_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_9_grad/StridedSliceGrad7gradients/lstm_1/strided_slice_10_grad/StridedSliceGrad7gradients/lstm_1/strided_slice_11_grad/StridedSliceGrad*)
_class
loc:@lstm_1/strided_slice_8*
T0*
_output_shapes
:*
N
ю
gradients/AddN_7AddN"gradients/lstm_1/Square_grad/mul_14gradients/lstm_1/strided_slice_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_1_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_2_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_3_grad/StridedSliceGrad* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:*
N
T
Const_2Const*
dtype0*
valueB*    *
_output_shapes
:
t
Variable
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:

Variable/AssignAssignVariableConst_2*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
:
e
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*
_output_shapes
:
\
Const_3Const*
dtype0*
valueB*    *
_output_shapes

:
~

Variable_1
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
Ё
Variable_1/AssignAssign
Variable_1Const_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes

:
T
Const_4Const*
dtype0*
valueB*    *
_output_shapes
:
v

Variable_2
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:

Variable_2/AssignAssign
Variable_2Const_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:
k
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes
:
\
Const_5Const*
dtype0*
valueB*    *
_output_shapes

:
~

Variable_3
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
Ё
Variable_3/AssignAssign
Variable_3Const_5*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes

:
o
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes

:
\
Const_6Const*
dtype0*
valueB*    *
_output_shapes

:
~

Variable_4
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
Ё
Variable_4/AssignAssign
Variable_4Const_6*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:
o
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes

:
J
mul_2Mulrho/readVariable/read*
T0*
_output_shapes
:
L
sub_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_1Subsub_1/xrho/read*
T0*
_output_shapes
: 
c
Square_1Square*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
B
mul_3Mulsub_1Square_1*
T0*
_output_shapes
:
?
add_2Addmul_2mul_3*
T0*
_output_shapes
:

AssignAssignVariableadd_2*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
:
f
mul_4Mullr/read*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
L
Const_7Const*
dtype0*
valueB
 *    *
_output_shapes
: 
L
Const_8Const*
dtype0*
valueB
 *  *
_output_shapes
: 
U
clip_by_value/MinimumMinimumadd_2Const_8*
T0*
_output_shapes
:
]
clip_by_valueMaximumclip_by_value/MinimumConst_7*
T0*
_output_shapes
:
@
SqrtSqrtclip_by_value*
T0*
_output_shapes
:
L
add_3/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
@
add_3AddSqrtadd_3/y*
T0*
_output_shapes
:
C
div_1RealDivmul_4add_3*
T0*
_output_shapes
:
K
sub_2Subdense_1/bias/readdiv_1*
T0*
_output_shapes
:

Assign_1Assigndense_1/biassub_2*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0*
_output_shapes
:
P
mul_5Mulrho/readVariable_1/read*
T0*
_output_shapes

:
L
sub_3/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_3Subsub_3/xrho/read*
T0*
_output_shapes
: 
c
Square_2Square&gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
F
mul_6Mulsub_3Square_2*
T0*
_output_shapes

:
C
add_4Addmul_5mul_6*
T0*
_output_shapes

:

Assign_2Assign
Variable_1add_4*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:
f
mul_7Mullr/read&gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
L
Const_9Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_10Const*
dtype0*
valueB
 *  *
_output_shapes
: 
\
clip_by_value_1/MinimumMinimumadd_4Const_10*
T0*
_output_shapes

:
e
clip_by_value_1Maximumclip_by_value_1/MinimumConst_9*
T0*
_output_shapes

:
H
Sqrt_1Sqrtclip_by_value_1*
T0*
_output_shapes

:
L
add_5/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
F
add_5AddSqrt_1add_5/y*
T0*
_output_shapes

:
G
div_2RealDivmul_7add_5*
T0*
_output_shapes

:
Q
sub_4Subdense_1/kernel/readdiv_2*
T0*
_output_shapes

:

Assign_3Assigndense_1/kernelsub_4*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:
L
mul_8Mulrho/readVariable_2/read*
T0*
_output_shapes
:
L
sub_5/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_5Subsub_5/xrho/read*
T0*
_output_shapes
: 
I
Square_3Squaregradients/AddN_6*
T0*
_output_shapes
:
B
mul_9Mulsub_5Square_3*
T0*
_output_shapes
:
?
add_6Addmul_8mul_9*
T0*
_output_shapes
:

Assign_4Assign
Variable_2add_6*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:
M
mul_10Mullr/readgradients/AddN_6*
T0*
_output_shapes
:
M
Const_11Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_12Const*
dtype0*
valueB
 *  *
_output_shapes
: 
X
clip_by_value_2/MinimumMinimumadd_6Const_12*
T0*
_output_shapes
:
b
clip_by_value_2Maximumclip_by_value_2/MinimumConst_11*
T0*
_output_shapes
:
D
Sqrt_2Sqrtclip_by_value_2*
T0*
_output_shapes
:
L
add_7/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
B
add_7AddSqrt_2add_7/y*
T0*
_output_shapes
:
D
div_3RealDivmul_10add_7*
T0*
_output_shapes
:
J
sub_6Sublstm_1/bias/readdiv_3*
T0*
_output_shapes
:

Assign_5Assignlstm_1/biassub_6*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes
:
Q
mul_11Mulrho/readVariable_3/read*
T0*
_output_shapes

:
L
sub_7/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_7Subsub_7/xrho/read*
T0*
_output_shapes
: 
M
Square_4Squaregradients/AddN_7*
T0*
_output_shapes

:
G
mul_12Mulsub_7Square_4*
T0*
_output_shapes

:
E
add_8Addmul_11mul_12*
T0*
_output_shapes

:

Assign_6Assign
Variable_3add_8*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes

:
Q
mul_13Mullr/readgradients/AddN_7*
T0*
_output_shapes

:
M
Const_13Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_14Const*
dtype0*
valueB
 *  *
_output_shapes
: 
\
clip_by_value_3/MinimumMinimumadd_8Const_14*
T0*
_output_shapes

:
f
clip_by_value_3Maximumclip_by_value_3/MinimumConst_13*
T0*
_output_shapes

:
H
Sqrt_3Sqrtclip_by_value_3*
T0*
_output_shapes

:
L
add_9/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
F
add_9AddSqrt_3add_9/y*
T0*
_output_shapes

:
H
div_4RealDivmul_13add_9*
T0*
_output_shapes

:
P
sub_8Sublstm_1/kernel/readdiv_4*
T0*
_output_shapes

:

Assign_7Assignlstm_1/kernelsub_8*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes

:
Q
mul_14Mulrho/readVariable_4/read*
T0*
_output_shapes

:
L
sub_9/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_9Subsub_9/xrho/read*
T0*
_output_shapes
: 
M
Square_5Squaregradients/AddN_5*
T0*
_output_shapes

:
G
mul_15Mulsub_9Square_5*
T0*
_output_shapes

:
F
add_10Addmul_14mul_15*
T0*
_output_shapes

:

Assign_8Assign
Variable_4add_10*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:
Q
mul_16Mullr/readgradients/AddN_5*
T0*
_output_shapes

:
M
Const_15Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_16Const*
dtype0*
valueB
 *  *
_output_shapes
: 
]
clip_by_value_4/MinimumMinimumadd_10Const_16*
T0*
_output_shapes

:
f
clip_by_value_4Maximumclip_by_value_4/MinimumConst_15*
T0*
_output_shapes

:
H
Sqrt_4Sqrtclip_by_value_4*
T0*
_output_shapes

:
M
add_11/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
H
add_11AddSqrt_4add_11/y*
T0*
_output_shapes

:
I
div_5RealDivmul_16add_11*
T0*
_output_shapes

:
[
sub_10Sublstm_1/recurrent_kernel/readdiv_5*
T0*
_output_shapes

:
Б
Assign_9Assignlstm_1/recurrent_kernelsub_10*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes

:

group_deps_1NoOp^add_1^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
Ѕ
initNoOp^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^lstm_1/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign
^lr/Assign^rho/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign"sХ.сo     лFыЗ	Ф}GжAJт
ќ-и-
9
Add
x"T
y"T
z"T"
Ttype:
2	
S
AddN
inputs"T*N
sum"T"
Nint(0"
Ttype:
2	
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
{
BiasAdd

value"T	
bias"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
{
BiasAddGrad
out_backprop"T
output"T"
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
8
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
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

ControlTrigger
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
y
Enter	
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

)
Exit	
data"T
output"T"	
Ttype
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
4
Fill
dims

value"T
output"T"	
Ttype
>
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
7
FloorMod
x"T
y"T
z"T"
Ttype:
2	
?
GreaterEqual
x"T
y"T
z
"
Ttype:
2		
.
Identity

input"T
output"T"	
Ttype
:
InvertPermutation
x"T
y"T"
Ttype0:
2	
7
Less
x"T
y"T
z
"
Ttype:
2		
<
	LessEqual
x"T
y"T
z
"
Ttype:
2		


LogicalNot
x

y

!
LoopCond	
input


output

o
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2
:
Maximum
x"T
y"T
z"T"
Ttype:	
2	

Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
:
Minimum
x"T
y"T
z"T"
Ttype:	
2	
<
Mul
x"T
y"T
z"T"
Ttype:
2	
-
Neg
x"T
y"T"
Ttype:
	2	
2
NextIteration	
data"T
output"T"	
Ttype

NoOp
D
NotEqual
x"T
y"T
z
"
Ttype:
2	

A
Placeholder
output"dtype"
dtypetype"
shapeshape: 

Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
}
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
`
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:
2	
=
RealDiv
x"T
y"T
z"T"
Ttype:
2	

RefEnter
data"T
output"T"	
Ttype"

frame_namestring"
is_constantbool( "
parallel_iterationsint

[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
e
ShapeN
input"T*N
output"out_type*N"
Nint(0"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
,
Sqrt
x"T
y"T"
Ttype:	
2
0
Square
x"T
y"T"
Ttype:
	2	
F
Stack
handle"
	elem_typetype"

stack_namestring 
?
StackPop
handle
elem"	elem_type"
	elem_typetype
V
	StackPush
handle	
elem"T
output"T"	
Ttype"
swap_memorybool( 
і
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 

StridedSliceGrad
shape"Index
begin"Index
end"Index
strides"Index
dy"T
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
5
Sub
x"T
y"T
z"T"
Ttype:
	2	

Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( "
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
,
Tanh
x"T
y"T"
Ttype:	
2
8
TanhGrad
x"T
y"T
z"T"
Ttype:	
2
x
TensorArrayGatherV3

handle
indices
flow_in
value"dtype"
dtypetype"
element_shapeshape:
`
TensorArrayGradV3

handle
flow_in
grad_handle
flow_out"
sourcestring
V
TensorArrayReadV3

handle	
index
flow_in
value"dtype"
dtypetype
a
TensorArrayScatterV3

handle
indices

value"T
flow_in
flow_out"	
Ttype
6
TensorArraySizeV3

handle
flow_in
size
И
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("
tensor_array_namestring 
]
TensorArrayWriteV3

handle	
index

value"T
flow_in
flow_out"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring 
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirty№Ъ
b
lstm_1_inputPlaceholder*
dtype0*
shape: *+
_output_shapes
:џџџџџџџџџ(
l
lstm_1/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
^
lstm_1/random_uniform/minConst*
dtype0*
valueB
 *7П*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
dtype0*
valueB
 *7?*
_output_shapes
: 
Ѕ
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
dtype0*
seed2с№i*
seedБџх)*
T0*
_output_shapes

:
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
T0*
_output_shapes
: 

lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
T0*
_output_shapes

:
{
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes

:

lstm_1/kernel
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
И
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes

:
x
lstm_1/kernel/readIdentitylstm_1/kernel* 
_class
loc:@lstm_1/kernel*
T0*
_output_shapes

:
T
lstm_1/SquareSquarelstm_1/kernel/read*
T0*
_output_shapes

:
Q
lstm_1/mul/xConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
W

lstm_1/mulMullstm_1/mul/xlstm_1/Square*
T0*
_output_shapes

:
]
lstm_1/ConstConst*
dtype0*
valueB"       *
_output_shapes
:
i

lstm_1/SumSum
lstm_1/mullstm_1/Const*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
Q
lstm_1/add/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
L

lstm_1/addAddlstm_1/add/x
lstm_1/Sum*
T0*
_output_shapes
: 

%lstm_1/recurrent_kernel/initial_valueConst*
dtype0*)
value B"ыyПФ.QПмП=шж<*
_output_shapes

:

lstm_1/recurrent_kernel
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
ц
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes

:

lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
T0*
_output_shapes

:
[
lstm_1/Const_1Const*
dtype0*
valueB*    *
_output_shapes
:
w
lstm_1/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
Ї
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/Const_1*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes
:
n
lstm_1/bias/readIdentitylstm_1/bias*
_class
loc:@lstm_1/bias*
T0*
_output_shapes
:
k
lstm_1/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
m
lstm_1/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
m
lstm_1/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Њ
lstm_1/strided_sliceStridedSlicelstm_1/kernel/readlstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_1/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
В
lstm_1/strided_slice_1StridedSlicelstm_1/kernel/readlstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_2/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
В
lstm_1/strided_slice_2StridedSlicelstm_1/kernel/readlstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_3/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_3/stack_1Const*
dtype0*
valueB"        *
_output_shapes
:
o
lstm_1/strided_slice_3/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
В
lstm_1/strided_slice_3StridedSlicelstm_1/kernel/readlstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_4/stackConst*
dtype0*
valueB"        *
_output_shapes
:
o
lstm_1/strided_slice_4/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_4/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
М
lstm_1/strided_slice_4StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_5/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
М
lstm_1/strided_slice_5StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_6/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
М
lstm_1/strided_slice_6StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_7/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_7/stack_1Const*
dtype0*
valueB"        *
_output_shapes
:
o
lstm_1/strided_slice_7/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
М
lstm_1/strided_slice_7StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
f
lstm_1/strided_slice_8/stackConst*
dtype0*
valueB: *
_output_shapes
:
h
lstm_1/strided_slice_8/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ќ
lstm_1/strided_slice_8StridedSlicelstm_1/bias/readlstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
f
lstm_1/strided_slice_9/stackConst*
dtype0*
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ќ
lstm_1/strided_slice_9StridedSlicelstm_1/bias/readlstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_10/stackConst*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
А
lstm_1/strided_slice_10StridedSlicelstm_1/bias/readlstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_11/stackConst*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_11/stack_1Const*
dtype0*
valueB: *
_output_shapes
:
i
lstm_1/strided_slice_11/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
А
lstm_1/strided_slice_11StridedSlicelstm_1/bias/readlstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
b
lstm_1/zeros_like	ZerosLikelstm_1_input*
T0*+
_output_shapes
:џџџџџџџџџ(
o
lstm_1/Sum_1/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:

lstm_1/Sum_1Sumlstm_1/zeros_likelstm_1/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ
`
lstm_1/ExpandDims/dimConst*
dtype0*
valueB :
џџџџџџџџџ*
_output_shapes
: 

lstm_1/ExpandDims
ExpandDimslstm_1/Sum_1lstm_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:џџџџџџџџџ
f
lstm_1/Tile/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:

lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:џџџџџџџџџ
e
lstm_1/Reshape/shapeConst*
dtype0*
valueB"џџџџ   *
_output_shapes
:
}
lstm_1/ReshapeReshapelstm_1_inputlstm_1/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMulMatMullstm_1/Reshapelstm_1/strided_slice*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/stackConst*
dtype0*!
valueB"џџџџ(      *
_output_shapes
:
}
lstm_1/Reshape_1Reshapelstm_1/BiasAddlstm_1/stack*
Tshape0*
T0*+
_output_shapes
:џџџџџџџџџ(
g
lstm_1/Reshape_2/shapeConst*
dtype0*
valueB"џџџџ   *
_output_shapes
:

lstm_1/Reshape_2Reshapelstm_1_inputlstm_1/Reshape_2/shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_1MatMullstm_1/Reshape_2lstm_1/strided_slice_1*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_1/stack_1Const*
dtype0*!
valueB"џџџџ(      *
_output_shapes
:

lstm_1/Reshape_3Reshapelstm_1/BiasAdd_1lstm_1/stack_1*
Tshape0*
T0*+
_output_shapes
:џџџџџџџџџ(
g
lstm_1/Reshape_4/shapeConst*
dtype0*
valueB"џџџџ   *
_output_shapes
:

lstm_1/Reshape_4Reshapelstm_1_inputlstm_1/Reshape_4/shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_2MatMullstm_1/Reshape_4lstm_1/strided_slice_2*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_1/stack_2Const*
dtype0*!
valueB"џџџџ(      *
_output_shapes
:

lstm_1/Reshape_5Reshapelstm_1/BiasAdd_2lstm_1/stack_2*
Tshape0*
T0*+
_output_shapes
:џџџџџџџџџ(
g
lstm_1/Reshape_6/shapeConst*
dtype0*
valueB"џџџџ   *
_output_shapes
:

lstm_1/Reshape_6Reshapelstm_1_inputlstm_1/Reshape_6/shape*
Tshape0*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_3MatMullstm_1/Reshape_6lstm_1/strided_slice_3*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ
c
lstm_1/stack_3Const*
dtype0*!
valueB"џџџџ(      *
_output_shapes
:

lstm_1/Reshape_7Reshapelstm_1/BiasAdd_3lstm_1/stack_3*
Tshape0*
T0*+
_output_shapes
:џџџџџџџџџ(
T
lstm_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
Р
lstm_1/concatConcatV2lstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7lstm_1/concat/axis*+
_output_shapes
:џџџџџџџџџ(*

Tidx0*
T0*
N
j
lstm_1/transpose/permConst*
dtype0*!
valueB"          *
_output_shapes
:

lstm_1/transpose	Transposelstm_1/concatlstm_1/transpose/perm*
Tperm0*
T0*+
_output_shapes
:(џџџџџџџџџ
\
lstm_1/ShapeShapelstm_1/transpose*
out_type0*
T0*
_output_shapes
:
g
lstm_1/strided_slice_12/stackConst*
dtype0*
valueB: *
_output_shapes
:
i
lstm_1/strided_slice_12/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_12/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Ј
lstm_1/strided_slice_12StridedSlicelstm_1/Shapelstm_1/strided_slice_12/stacklstm_1/strided_slice_12/stack_1lstm_1/strided_slice_12/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
g
lstm_1/strided_slice_13/stackConst*
dtype0*
valueB: *
_output_shapes
:
i
lstm_1/strided_slice_13/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
i
lstm_1/strided_slice_13/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
Н
lstm_1/strided_slice_13StridedSlicelstm_1/transposelstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
n
lstm_1/strided_slice_14/stackConst*
dtype0*
valueB"        *
_output_shapes
:
p
lstm_1/strided_slice_14/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_14/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ф
lstm_1/strided_slice_14StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_14/stacklstm_1/strided_slice_14/stack_1lstm_1/strided_slice_14/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_15/stackConst*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ф
lstm_1/strided_slice_15StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_15/stacklstm_1/strided_slice_15/stack_1lstm_1/strided_slice_15/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_16/stackConst*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ф
lstm_1/strided_slice_16StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_16/stacklstm_1/strided_slice_16/stack_1lstm_1/strided_slice_16/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_17/stackConst*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_17/stack_1Const*
dtype0*
valueB"        *
_output_shapes
:
p
lstm_1/strided_slice_17/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
Ф
lstm_1/strided_slice_17StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_17/stacklstm_1/strided_slice_17/stack_1lstm_1/strided_slice_17/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
S
lstm_1/mul_1/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
lstm_1/mul_1Mullstm_1/Tilelstm_1/mul_1/y*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_4MatMullstm_1/mul_1lstm_1/strided_slice_4*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/add_1Addlstm_1/strided_slice_14lstm_1/MatMul_4*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_2/xConst*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
c
lstm_1/mul_2Mullstm_1/mul_2/xlstm_1/add_1*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_2/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
c
lstm_1/add_2Addlstm_1/mul_2lstm_1/add_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_2Const*
dtype0*
valueB
 *    *
_output_shapes
: 
S
lstm_1/Const_3Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_2lstm_1/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_2*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_3/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
lstm_1/mul_3Mullstm_1/Tilelstm_1/mul_3/y*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_5MatMullstm_1/mul_3lstm_1/strided_slice_5*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/add_3Addlstm_1/strided_slice_15lstm_1/MatMul_5*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_4/xConst*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
c
lstm_1/mul_4Mullstm_1/mul_4/xlstm_1/add_3*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_4/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
c
lstm_1/add_4Addlstm_1/mul_4lstm_1/add_4/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_4Const*
dtype0*
valueB
 *    *
_output_shapes
: 
S
lstm_1/Const_5Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
y
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/add_4lstm_1/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ
j
lstm_1/mul_5Mullstm_1/clip_by_value_1lstm_1/Tile*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_6/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
lstm_1/mul_6Mullstm_1/Tilelstm_1/mul_6/y*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_6MatMullstm_1/mul_6lstm_1/strided_slice_6*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/add_5Addlstm_1/strided_slice_16lstm_1/MatMul_6*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/TanhTanhlstm_1/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
h
lstm_1/mul_7Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/add_6Addlstm_1/mul_5lstm_1/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_8/yConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
b
lstm_1/mul_8Mullstm_1/Tilelstm_1/mul_8/y*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/MatMul_7MatMullstm_1/mul_8lstm_1/strided_slice_7*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/add_7Addlstm_1/strided_slice_17lstm_1/MatMul_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/mul_9/xConst*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
c
lstm_1/mul_9Mullstm_1/mul_9/xlstm_1/add_7*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/add_8/yConst*
dtype0*
valueB
 *   ?*
_output_shapes
: 
c
lstm_1/add_8Addlstm_1/mul_9lstm_1/add_8/y*
T0*'
_output_shapes
:џџџџџџџџџ
S
lstm_1/Const_6Const*
dtype0*
valueB
 *    *
_output_shapes
: 
S
lstm_1/Const_7Const*
dtype0*
valueB
 *  ?*
_output_shapes
: 
y
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/add_8lstm_1/Const_7*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_6*
T0*'
_output_shapes
:џџџџџџџџџ
U
lstm_1/Tanh_1Tanhlstm_1/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
m
lstm_1/mul_10Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ю
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_12*
_output_shapes

::*
dtype0*
dynamic_size( *
clear_after_read(* 
tensor_array_name	output_ta*
element_shape:
Я
lstm_1/TensorArray_1TensorArrayV3lstm_1/strided_slice_12*
_output_shapes

::*
dtype0*
dynamic_size( *
clear_after_read(*
tensor_array_name
input_ta*
element_shape:
o
lstm_1/TensorArrayUnstack/ShapeShapelstm_1/transpose*
out_type0*
T0*
_output_shapes
:
w
-lstm_1/TensorArrayUnstack/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_1Const*
dtype0*
valueB:*
_output_shapes
:
y
/lstm_1/TensorArrayUnstack/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
ћ
'lstm_1/TensorArrayUnstack/strided_sliceStridedSlicelstm_1/TensorArrayUnstack/Shape-lstm_1/TensorArrayUnstack/strided_slice/stack/lstm_1/TensorArrayUnstack/strided_slice/stack_1/lstm_1/TensorArrayUnstack/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes
: *

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask
g
%lstm_1/TensorArrayUnstack/range/startConst*
dtype0*
value	B : *
_output_shapes
: 
g
%lstm_1/TensorArrayUnstack/range/deltaConst*
dtype0*
value	B :*
_output_shapes
: 
а
lstm_1/TensorArrayUnstack/rangeRange%lstm_1/TensorArrayUnstack/range/start'lstm_1/TensorArrayUnstack/strided_slice%lstm_1/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:џџџџџџџџџ

Alstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3TensorArrayScatterV3lstm_1/TensorArray_1lstm_1/TensorArrayUnstack/rangelstm_1/transposelstm_1/TensorArray_1:1*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
M
lstm_1/timeConst*
dtype0*
value	B : *
_output_shapes
: 
Ѓ
lstm_1/while/EnterEnterlstm_1/time*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes
: 
А
lstm_1/while/Enter_1Enterlstm_1/TensorArray:1*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes
:
Ж
lstm_1/while/Enter_2Enterlstm_1/Tile*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*'
_output_shapes
:џџџџџџџџџ
Ж
lstm_1/while/Enter_3Enterlstm_1/Tile*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*'
_output_shapes
:џџџџџџџџџ
w
lstm_1/while/MergeMergelstm_1/while/Enterlstm_1/while/NextIteration*
N*
T0*
_output_shapes
: : 

lstm_1/while/Merge_1Mergelstm_1/while/Enter_1lstm_1/while/NextIteration_1*
N*
T0*
_output_shapes
:: 

lstm_1/while/Merge_2Mergelstm_1/while/Enter_2lstm_1/while/NextIteration_2*
N*
T0*)
_output_shapes
:џџџџџџџџџ: 

lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*
N*
T0*)
_output_shapes
:џџџџџџџџџ: 
Д
lstm_1/while/Less/EnterEnterlstm_1/strided_slice_12*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes
: 
g
lstm_1/while/LessLesslstm_1/while/Mergelstm_1/while/Less/Enter*
T0*
_output_shapes
: 
L
lstm_1/while/LoopCondLoopCondlstm_1/while/Less*
_output_shapes
: 

lstm_1/while/SwitchSwitchlstm_1/while/Mergelstm_1/while/LoopCond*%
_class
loc:@lstm_1/while/Merge*
T0*
_output_shapes
: : 

lstm_1/while/Switch_1Switchlstm_1/while/Merge_1lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes

::
К
lstm_1/while/Switch_2Switchlstm_1/while/Merge_2lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_2*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
К
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
Y
lstm_1/while/IdentityIdentitylstm_1/while/Switch:1*
T0*
_output_shapes
: 
_
lstm_1/while/Identity_1Identitylstm_1/while/Switch_1:1*
T0*
_output_shapes
:
n
lstm_1/while/Identity_2Identitylstm_1/while/Switch_2:1*
T0*'
_output_shapes
:џџџџџџџџџ
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*
T0*'
_output_shapes
:џџџџџџџџџ
ы
$lstm_1/while/TensorArrayReadV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(

&lstm_1/while/TensorArrayReadV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: **

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
ї
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity&lstm_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*'
_output_shapes
:џџџџџџџџџ

 lstm_1/while/strided_slice/stackConst^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:

"lstm_1/while/strided_slice/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

"lstm_1/while/strided_slice/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
з
lstm_1/while/strided_sliceStridedSlicelstm_1/while/TensorArrayReadV3 lstm_1/while/strided_slice/stack"lstm_1/while/strided_slice/stack_1"lstm_1/while/strided_slice/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 

"lstm_1/while/strided_slice_1/stackConst^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_1/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_1/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
п
lstm_1/while/strided_slice_1StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_1/stack$lstm_1/while/strided_slice_1/stack_1$lstm_1/while/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 

"lstm_1/while/strided_slice_2/stackConst^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_2/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_2/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
п
lstm_1/while/strided_slice_2StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_2/stack$lstm_1/while/strided_slice_2/stack_1$lstm_1/while/strided_slice_2/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 

"lstm_1/while/strided_slice_3/stackConst^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:

$lstm_1/while/strided_slice_3/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:

$lstm_1/while/strided_slice_3/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
п
lstm_1/while/strided_slice_3StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_3/stack$lstm_1/while/strided_slice_3/stack_1$lstm_1/while/strided_slice_3/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
o
lstm_1/while/mul/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
v
lstm_1/while/mulMullstm_1/while/Identity_2lstm_1/while/mul/y*
T0*'
_output_shapes
:џџџџџџџџџ
Н
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice_4*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:
Ђ
lstm_1/while/MatMulMatMullstm_1/while/mullstm_1/while/MatMul/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ
z
lstm_1/while/addAddlstm_1/while/strided_slicelstm_1/while/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_1/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
s
lstm_1/while/mul_1Mullstm_1/while/mul_1/xlstm_1/while/add*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_1/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *   ?*
_output_shapes
: 
u
lstm_1/while/add_1Addlstm_1/while/mul_1lstm_1/while/add_1/y*
T0*'
_output_shapes
:џџџџџџџџџ
o
lstm_1/while/ConstConst^lstm_1/while/Identity*
dtype0*
valueB
 *    *
_output_shapes
: 
q
lstm_1/while/Const_1Const^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 

"lstm_1/while/clip_by_value/MinimumMinimumlstm_1/while/add_1lstm_1/while/Const_1*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_2/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
z
lstm_1/while/mul_2Mullstm_1/while/Identity_2lstm_1/while/mul_2/y*
T0*'
_output_shapes
:џџџџџџџџџ
П
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_5*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:
Ј
lstm_1/while/MatMul_1MatMullstm_1/while/mul_2lstm_1/while/MatMul_1/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/add_2Addlstm_1/while/strided_slice_1lstm_1/while/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_3/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
u
lstm_1/while/mul_3Mullstm_1/while/mul_3/xlstm_1/while/add_2*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_3/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *   ?*
_output_shapes
: 
u
lstm_1/while/add_3Addlstm_1/while/mul_3lstm_1/while/add_3/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/Const_2Const^lstm_1/while/Identity*
dtype0*
valueB
 *    *
_output_shapes
: 
q
lstm_1/while/Const_3Const^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_3lstm_1/while/Const_3*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/mul_4Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_3*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_5/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
z
lstm_1/while/mul_5Mullstm_1/while/Identity_2lstm_1/while/mul_5/y*
T0*'
_output_shapes
:џџџџџџџџџ
П
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_6*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:
Ј
lstm_1/while/MatMul_2MatMullstm_1/while/mul_5lstm_1/while/MatMul_2/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/add_4Addlstm_1/while/strided_slice_2lstm_1/while/MatMul_2*
T0*'
_output_shapes
:џџџџџџџџџ
_
lstm_1/while/TanhTanhlstm_1/while/add_4*
T0*'
_output_shapes
:џџџџџџџџџ
z
lstm_1/while/mul_6Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
s
lstm_1/while/add_5Addlstm_1/while/mul_4lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_7/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 
z
lstm_1/while/mul_7Mullstm_1/while/Identity_2lstm_1/while/mul_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
П
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_7*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:
Ј
lstm_1/while/MatMul_3MatMullstm_1/while/mul_7lstm_1/while/MatMul_3/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/add_6Addlstm_1/while/strided_slice_3lstm_1/while/MatMul_3*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/mul_8/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *ЭЬL>*
_output_shapes
: 
u
lstm_1/while/mul_8Mullstm_1/while/mul_8/xlstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/add_7/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *   ?*
_output_shapes
: 
u
lstm_1/while/add_7Addlstm_1/while/mul_8lstm_1/while/add_7/y*
T0*'
_output_shapes
:џџџџџџџџџ
q
lstm_1/while/Const_4Const^lstm_1/while/Identity*
dtype0*
valueB
 *    *
_output_shapes
: 
q
lstm_1/while/Const_5Const^lstm_1/while/Identity*
dtype0*
valueB
 *  ?*
_output_shapes
: 

$lstm_1/while/clip_by_value_2/MinimumMinimumlstm_1/while/add_7lstm_1/while/Const_5*
T0*'
_output_shapes
:џџџџџџџџџ

lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:џџџџџџџџџ
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
~
lstm_1/while/mul_9Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
љ
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/TensorArray*
is_constant(

0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV36lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enterlstm_1/while/Identitylstm_1/while/mul_9lstm_1/while/Identity_1*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
: 
n
lstm_1/while/add_8/yConst^lstm_1/while/Identity*
dtype0*
value	B :*
_output_shapes
: 
g
lstm_1/while/add_8Addlstm_1/while/Identitylstm_1/while/add_8/y*
T0*
_output_shapes
: 
`
lstm_1/while/NextIterationNextIterationlstm_1/while/add_8*
T0*
_output_shapes
: 

lstm_1/while/NextIteration_1NextIteration0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
s
lstm_1/while/NextIteration_2NextIterationlstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
O
lstm_1/while/ExitExitlstm_1/while/Switch*
T0*
_output_shapes
: 
U
lstm_1/while/Exit_1Exitlstm_1/while/Switch_1*
T0*
_output_shapes
:
d
lstm_1/while/Exit_2Exitlstm_1/while/Switch_2*
T0*'
_output_shapes
:џџџџџџџџџ
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*
T0*'
_output_shapes
:џџџџџџџџџ
І
)lstm_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm_1/TensorArraylstm_1/while/Exit_1*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 

#lstm_1/TensorArrayStack/range/startConst*
dtype0*%
_class
loc:@lstm_1/TensorArray*
value	B : *
_output_shapes
: 

#lstm_1/TensorArrayStack/range/deltaConst*
dtype0*%
_class
loc:@lstm_1/TensorArray*
value	B :*
_output_shapes
: 
ѓ
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*%
_class
loc:@lstm_1/TensorArray*

Tidx0*#
_output_shapes
:џџџџџџџџџ

+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_1*$
element_shape:џџџџџџџџџ*
dtype0*%
_class
loc:@lstm_1/TensorArray*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
N
lstm_1/sub/yConst*
dtype0*
value	B :*
_output_shapes
: 
S

lstm_1/subSublstm_1/while/Exitlstm_1/sub/y*
T0*
_output_shapes
: 
П
lstm_1/TensorArrayReadV3TensorArrayReadV3lstm_1/TensorArray
lstm_1/sublstm_1/while/Exit_1*
dtype0*%
_class
loc:@lstm_1/TensorArray*'
_output_shapes
:џџџџџџџџџ
l
lstm_1/transpose_1/permConst*
dtype0*!
valueB"          *
_output_shapes
:
Б
lstm_1/transpose_1	Transpose+lstm_1/TensorArrayStack/TensorArrayGatherV3lstm_1/transpose_1/perm*
Tperm0*
T0*4
_output_shapes"
 :џџџџџџџџџџџџџџџџџџ
e
lstm_1/Square_1Squarelstm_1/TensorArrayReadV3*
T0*'
_output_shapes
:џџџџџџџџџ
T
lstm_1/mul_11/xConst*
dtype0*
valueB
 *o:*
_output_shapes
: 
h
lstm_1/mul_11Mullstm_1/mul_11/xlstm_1/Square_1*
T0*'
_output_shapes
:џџџџџџџџџ
_
lstm_1/Const_8Const*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/Sum_2Sumlstm_1/mul_11lstm_1/Const_8*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
S
lstm_1/add_9/xConst*
dtype0*
valueB
 *    *
_output_shapes
: 
R
lstm_1/add_9Addlstm_1/add_9/xlstm_1/Sum_2*
T0*
_output_shapes
: 
m
dense_1/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
valueB
 *зГнП*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
dtype0*
valueB
 *зГн?*
_output_shapes
: 
Ј
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2е*
seedБџх)*
T0*
_output_shapes

:
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 

dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:

dense_1/kernel
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
М
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:
Z
dense_1/ConstConst*
dtype0*
valueB*    *
_output_shapes
:
x
dense_1/bias
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:
Љ
dense_1/bias/AssignAssigndense_1/biasdense_1/Const*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0*
_output_shapes
:
q
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
T0*
_output_shapes
:

dense_1/MatMulMatMullstm_1/TensorArrayReadV3dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:џџџџџџџџџ

dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:џџџџџџџџџ
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:џџџџџџџџџ
U
lr/initial_valueConst*
dtype0*
valueB
 *
з#<*
_output_shapes
: 
f
lr
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 

	lr/AssignAssignlrlr/initial_value*
validate_shape(*
_class
	loc:@lr*
use_locking(*
T0*
_output_shapes
: 
O
lr/readIdentitylr*
_class
	loc:@lr*
T0*
_output_shapes
: 
V
rho/initial_valueConst*
dtype0*
valueB
 *fff?*
_output_shapes
: 
g
rho
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 


rho/AssignAssignrhorho/initial_value*
validate_shape(*
_class

loc:@rho*
use_locking(*
T0*
_output_shapes
: 
R
rho/readIdentityrho*
_class

loc:@rho*
T0*
_output_shapes
: 
X
decay/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
i
decay
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 

decay/AssignAssigndecaydecay/initial_value*
validate_shape(*
_class

loc:@decay*
use_locking(*
T0*
_output_shapes
: 
X

decay/readIdentitydecay*
_class

loc:@decay*
T0*
_output_shapes
: 
]
iterations/initial_valueConst*
dtype0*
valueB
 *    *
_output_shapes
: 
n

iterations
VariableV2*
dtype0*
shape: *
shared_name *
	container *
_output_shapes
: 
Њ
iterations/AssignAssign
iterationsiterations/initial_value*
validate_shape(*
_class
loc:@iterations*
use_locking(*
T0*
_output_shapes
: 
g
iterations/readIdentity
iterations*
_class
loc:@iterations*
T0*
_output_shapes
: 
d
dense_1_sample_weightsPlaceholder*
dtype0*
shape: *#
_output_shapes
:џџџџџџџџџ
i
dense_1_targetPlaceholder*
dtype0*
shape: *0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
c
subSubdense_1/Tanhdense_1_target*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
P
SquareSquaresub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
X
Mean/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
w
MeanMeanSquareMean/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ
[
Mean_1/reduction_indicesConst*
dtype0*
valueB *
_output_shapes
: 
y
Mean_1MeanMeanMean_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:џџџџџџџџџ
X
mulMulMean_1dense_1_sample_weights*
T0*#
_output_shapes
:џџџџџџџџџ
O

NotEqual/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
f
NotEqualNotEqualdense_1_sample_weights
NotEqual/y*
T0*#
_output_shapes
:џџџџџџџџџ
S
CastCastNotEqual*

DstT0*

SrcT0
*#
_output_shapes
:џџџџџџџџџ
O
ConstConst*
dtype0*
valueB: *
_output_shapes
:
Y
Mean_2MeanCastConst*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
I
divRealDivmulMean_2*
T0*#
_output_shapes
:џџџџџџџџџ
Q
Const_1Const*
dtype0*
valueB: *
_output_shapes
:
Z
Mean_3MeandivConst_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
L
mul_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
>
mul_1Mulmul_1/xMean_3*
T0*
_output_shapes
: 
>
addAddmul_1
lstm_1/add*
T0*
_output_shapes
: 
@
add_1Addaddlstm_1/add_9*
T0*
_output_shapes
: 


group_depsNoOp^add_1
l
gradients/ShapeConst*
dtype0*
_class

loc:@add_1*
valueB *
_output_shapes
: 
n
gradients/ConstConst*
dtype0*
_class

loc:@add_1*
valueB
 *  ?*
_output_shapes
: 
s
gradients/FillFillgradients/Shapegradients/Const*
_class

loc:@add_1*
T0*
_output_shapes
: 
{
gradients/f_countConst*
dtype0*&
_class
loc:@lstm_1/while/Exit_1*
value	B : *
_output_shapes
: 
в
gradients/f_count_1Entergradients/f_count*
_output_shapes
: **

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 

gradients/MergeMergegradients/f_count_1gradients/NextIteration*
N*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 

gradients/SwitchSwitchgradients/Mergelstm_1/while/LoopCond*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 

gradients/Add/yConst^lstm_1/while/Identity*
dtype0*&
_class
loc:@lstm_1/while/Exit_1*
value	B :*
_output_shapes
: 

gradients/AddAddgradients/Switch:1gradients/Add/y*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
њ+
gradients/NextIterationNextIterationgradients/Add\^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPushB^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush_10^gradients/lstm_1/while/mul_9_grad/mul/StackPush2^gradients/lstm_1/while/mul_9_grad/mul_1/StackPushC^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPushE^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush_1L^gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPushH^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPushJ^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush_1T^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush_1B^gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush_10^gradients/lstm_1/while/mul_4_grad/mul/StackPush2^gradients/lstm_1/while/mul_4_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush_10^gradients/lstm_1/while/mul_6_grad/mul/StackPush2^gradients/lstm_1/while/mul_6_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_8_grad/mul/StackPush2^gradients/lstm_1/while/mul_8_grad/mul_1/StackPushC^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPushE^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush_1L^gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPushA^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPushC^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush_1J^gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush_1H^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPushJ^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush_1T^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPushF^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPushH^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush_1R^gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush_1G^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPushI^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_1I^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_2I^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_38^gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPushB^gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPushG^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPushI^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_1I^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_2I^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_38^gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPushB^gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_7_grad/mul/StackPush2^gradients/lstm_1/while/mul_7_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_3_grad/mul/StackPush2^gradients/lstm_1/while/mul_3_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_1_grad/mul/StackPush2^gradients/lstm_1/while/mul_1_grad/mul_1/StackPushB^gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_5_grad/mul/StackPushB^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPushD^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush_1@^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPushB^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush_1G^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPushI^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_1I^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_2I^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_38^gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPushE^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPushG^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_1G^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_2G^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_36^gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPushB^gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPush0^gradients/lstm_1/while/mul_2_grad/mul/StackPush@^gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPush.^gradients/lstm_1/while/mul_grad/mul/StackPush*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
v
gradients/f_count_2Exitgradients/Switch*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
{
gradients/b_countConst*
dtype0*&
_class
loc:@lstm_1/while/Exit_1*
value	B :*
_output_shapes
: 
о
gradients/b_count_1Entergradients/f_count_2*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 

gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
N*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 
х
gradients/GreaterEqual/EnterEntergradients/b_count*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant(
 
gradients/GreaterEqualGreaterEqualgradients/Merge_1gradients/GreaterEqual/Enter*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
w
gradients/b_count_2LoopCondgradients/GreaterEqual*&
_class
loc:@lstm_1/while/Exit_1*
_output_shapes
: 

gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 

gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
н
gradients/NextIteration_1NextIterationgradients/SubY^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_sync*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
x
gradients/b_count_3Exitgradients/Switch_1*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
w
gradients/add_1_grad/ShapeConst*
dtype0*
_class

loc:@add_1*
valueB *
_output_shapes
: 
y
gradients/add_1_grad/Shape_1Const*
dtype0*
_class

loc:@add_1*
valueB *
_output_shapes
: 
д
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
_class

loc:@add_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Е
gradients/add_1_grad/SumSumgradients/Fill*gradients/add_1_grad/BroadcastGradientArgs*
_class

loc:@add_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
І
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
_class

loc:@add_1*
T0*
_output_shapes
: 
Й
gradients/add_1_grad/Sum_1Sumgradients/Fill,gradients/add_1_grad/BroadcastGradientArgs:1*
_class

loc:@add_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ќ
gradients/add_1_grad/Reshape_1Reshapegradients/add_1_grad/Sum_1gradients/add_1_grad/Shape_1*
Tshape0*
_class

loc:@add_1*
T0*
_output_shapes
: 
s
gradients/add_grad/ShapeConst*
dtype0*
_class

loc:@add*
valueB *
_output_shapes
: 
u
gradients/add_grad/Shape_1Const*
dtype0*
_class

loc:@add*
valueB *
_output_shapes
: 
Ь
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
_class

loc:@add*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Н
gradients/add_grad/SumSumgradients/add_1_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
_class

loc:@add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
_class

loc:@add*
T0*
_output_shapes
: 
С
gradients/add_grad/Sum_1Sumgradients/add_1_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
_class

loc:@add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Є
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_class

loc:@add*
T0*
_output_shapes
: 

!gradients/lstm_1/add_9_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/add_9*
valueB *
_output_shapes
: 

#gradients/lstm_1/add_9_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/add_9*
valueB *
_output_shapes
: 
№
1gradients/lstm_1/add_9_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/lstm_1/add_9_grad/Shape#gradients/lstm_1/add_9_grad/Shape_1*
_class
loc:@lstm_1/add_9*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
к
gradients/lstm_1/add_9_grad/SumSumgradients/add_1_grad/Reshape_11gradients/lstm_1/add_9_grad/BroadcastGradientArgs*
_class
loc:@lstm_1/add_9*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Т
#gradients/lstm_1/add_9_grad/ReshapeReshapegradients/lstm_1/add_9_grad/Sum!gradients/lstm_1/add_9_grad/Shape*
Tshape0*
_class
loc:@lstm_1/add_9*
T0*
_output_shapes
: 
о
!gradients/lstm_1/add_9_grad/Sum_1Sumgradients/add_1_grad/Reshape_13gradients/lstm_1/add_9_grad/BroadcastGradientArgs:1*
_class
loc:@lstm_1/add_9*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ш
%gradients/lstm_1/add_9_grad/Reshape_1Reshape!gradients/lstm_1/add_9_grad/Sum_1#gradients/lstm_1/add_9_grad/Shape_1*
Tshape0*
_class
loc:@lstm_1/add_9*
T0*
_output_shapes
: 
w
gradients/mul_1_grad/ShapeConst*
dtype0*
_class

loc:@mul_1*
valueB *
_output_shapes
: 
y
gradients/mul_1_grad/Shape_1Const*
dtype0*
_class

loc:@mul_1*
valueB *
_output_shapes
: 
д
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
~
gradients/mul_1_grad/mulMulgradients/add_grad/ReshapeMean_3*
_class

loc:@mul_1*
T0*
_output_shapes
: 
П
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_class

loc:@mul_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
І
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
Tshape0*
_class

loc:@mul_1*
T0*
_output_shapes
: 

gradients/mul_1_grad/mul_1Mulmul_1/xgradients/add_grad/Reshape*
_class

loc:@mul_1*
T0*
_output_shapes
: 
Х
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_class

loc:@mul_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ќ
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
Tshape0*
_class

loc:@mul_1*
T0*
_output_shapes
: 

gradients/lstm_1/add_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/add*
valueB *
_output_shapes
: 

!gradients/lstm_1/add_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/add*
valueB *
_output_shapes
: 
ш
/gradients/lstm_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/lstm_1/add_grad/Shape!gradients/lstm_1/add_grad/Shape_1*
_class
loc:@lstm_1/add*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
в
gradients/lstm_1/add_grad/SumSumgradients/add_grad/Reshape_1/gradients/lstm_1/add_grad/BroadcastGradientArgs*
_class
loc:@lstm_1/add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
К
!gradients/lstm_1/add_grad/ReshapeReshapegradients/lstm_1/add_grad/Sumgradients/lstm_1/add_grad/Shape*
Tshape0*
_class
loc:@lstm_1/add*
T0*
_output_shapes
: 
ж
gradients/lstm_1/add_grad/Sum_1Sumgradients/add_grad/Reshape_11gradients/lstm_1/add_grad/BroadcastGradientArgs:1*
_class
loc:@lstm_1/add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Р
#gradients/lstm_1/add_grad/Reshape_1Reshapegradients/lstm_1/add_grad/Sum_1!gradients/lstm_1/add_grad/Shape_1*
Tshape0*
_class
loc:@lstm_1/add*
T0*
_output_shapes
: 

)gradients/lstm_1/Sum_2_grad/Reshape/shapeConst*
dtype0*
_class
loc:@lstm_1/Sum_2*
valueB"      *
_output_shapes
:
и
#gradients/lstm_1/Sum_2_grad/ReshapeReshape%gradients/lstm_1/add_9_grad/Reshape_1)gradients/lstm_1/Sum_2_grad/Reshape/shape*
Tshape0*
_class
loc:@lstm_1/Sum_2*
T0*
_output_shapes

:

!gradients/lstm_1/Sum_2_grad/ShapeShapelstm_1/mul_11*
_class
loc:@lstm_1/Sum_2*
out_type0*
T0*
_output_shapes
:
е
 gradients/lstm_1/Sum_2_grad/TileTile#gradients/lstm_1/Sum_2_grad/Reshape!gradients/lstm_1/Sum_2_grad/Shape*

Tmultiples0*
_class
loc:@lstm_1/Sum_2*
T0*'
_output_shapes
:џџџџџџџџџ

#gradients/Mean_3_grad/Reshape/shapeConst*
dtype0*
_class
loc:@Mean_3*
valueB:*
_output_shapes
:
Л
gradients/Mean_3_grad/ReshapeReshapegradients/mul_1_grad/Reshape_1#gradients/Mean_3_grad/Reshape/shape*
Tshape0*
_class
loc:@Mean_3*
T0*
_output_shapes
:
y
gradients/Mean_3_grad/ShapeShapediv*
_class
loc:@Mean_3*
out_type0*
T0*
_output_shapes
:
Й
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*

Tmultiples0*
_class
loc:@Mean_3*
T0*#
_output_shapes
:џџџџџџџџџ
{
gradients/Mean_3_grad/Shape_1Shapediv*
_class
loc:@Mean_3*
out_type0*
T0*
_output_shapes
:
{
gradients/Mean_3_grad/Shape_2Const*
dtype0*
_class
loc:@Mean_3*
valueB *
_output_shapes
: 

gradients/Mean_3_grad/ConstConst*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
З
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
_class
loc:@Mean_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

gradients/Mean_3_grad/Const_1Const*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
Л
gradients/Mean_3_grad/Prod_1Prodgradients/Mean_3_grad/Shape_2gradients/Mean_3_grad/Const_1*
_class
loc:@Mean_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
|
gradients/Mean_3_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_3*
value	B :*
_output_shapes
: 
Ѓ
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
Ё
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
_class
loc:@Mean_3*
T0*
_output_shapes
: 

gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

DstT0*
_class
loc:@Mean_3*

SrcT0*
_output_shapes
: 
Љ
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
_class
loc:@Mean_3*
T0*#
_output_shapes
:џџџџџџџџџ

'gradients/lstm_1/Sum_grad/Reshape/shapeConst*
dtype0*
_class
loc:@lstm_1/Sum*
valueB"      *
_output_shapes
:
а
!gradients/lstm_1/Sum_grad/ReshapeReshape#gradients/lstm_1/add_grad/Reshape_1'gradients/lstm_1/Sum_grad/Reshape/shape*
Tshape0*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes

:

(gradients/lstm_1/Sum_grad/Tile/multiplesConst*
dtype0*
_class
loc:@lstm_1/Sum*
valueB"      *
_output_shapes
:
Э
gradients/lstm_1/Sum_grad/TileTile!gradients/lstm_1/Sum_grad/Reshape(gradients/lstm_1/Sum_grad/Tile/multiples*

Tmultiples0*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes

:

"gradients/lstm_1/mul_11_grad/ShapeConst*
dtype0* 
_class
loc:@lstm_1/mul_11*
valueB *
_output_shapes
: 

$gradients/lstm_1/mul_11_grad/Shape_1Shapelstm_1/Square_1* 
_class
loc:@lstm_1/mul_11*
out_type0*
T0*
_output_shapes
:
є
2gradients/lstm_1/mul_11_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/lstm_1/mul_11_grad/Shape$gradients/lstm_1/mul_11_grad/Shape_1* 
_class
loc:@lstm_1/mul_11*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Ў
 gradients/lstm_1/mul_11_grad/mulMul gradients/lstm_1/Sum_2_grad/Tilelstm_1/Square_1* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:џџџџџџџџџ
п
 gradients/lstm_1/mul_11_grad/SumSum gradients/lstm_1/mul_11_grad/mul2gradients/lstm_1/mul_11_grad/BroadcastGradientArgs* 
_class
loc:@lstm_1/mul_11*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ц
$gradients/lstm_1/mul_11_grad/ReshapeReshape gradients/lstm_1/mul_11_grad/Sum"gradients/lstm_1/mul_11_grad/Shape*
Tshape0* 
_class
loc:@lstm_1/mul_11*
T0*
_output_shapes
: 
А
"gradients/lstm_1/mul_11_grad/mul_1Mullstm_1/mul_11/x gradients/lstm_1/Sum_2_grad/Tile* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:џџџџџџџџџ
х
"gradients/lstm_1/mul_11_grad/Sum_1Sum"gradients/lstm_1/mul_11_grad/mul_14gradients/lstm_1/mul_11_grad/BroadcastGradientArgs:1* 
_class
loc:@lstm_1/mul_11*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
н
&gradients/lstm_1/mul_11_grad/Reshape_1Reshape"gradients/lstm_1/mul_11_grad/Sum_1$gradients/lstm_1/mul_11_grad/Shape_1*
Tshape0* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:џџџџџџџџџ
s
gradients/div_grad/ShapeShapemul*
_class

loc:@div*
out_type0*
T0*
_output_shapes
:
u
gradients/div_grad/Shape_1Const*
dtype0*
_class

loc:@div*
valueB *
_output_shapes
: 
Ь
(gradients/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_grad/Shapegradients/div_grad/Shape_1*
_class

loc:@div*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/div_grad/RealDivRealDivgradients/Mean_3_grad/truedivMean_2*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ
Л
gradients/div_grad/SumSumgradients/div_grad/RealDiv(gradients/div_grad/BroadcastGradientArgs*
_class

loc:@div*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ћ
gradients/div_grad/ReshapeReshapegradients/div_grad/Sumgradients/div_grad/Shape*
Tshape0*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ
h
gradients/div_grad/NegNegmul*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/div_grad/RealDiv_1RealDivgradients/div_grad/NegMean_2*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/div_grad/RealDiv_2RealDivgradients/div_grad/RealDiv_1Mean_2*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ
 
gradients/div_grad/mulMulgradients/Mean_3_grad/truedivgradients/div_grad/RealDiv_2*
_class

loc:@div*
T0*#
_output_shapes
:џџџџџџџџџ
Л
gradients/div_grad/Sum_1Sumgradients/div_grad/mul*gradients/div_grad/BroadcastGradientArgs:1*
_class

loc:@div*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Є
gradients/div_grad/Reshape_1Reshapegradients/div_grad/Sum_1gradients/div_grad/Shape_1*
Tshape0*
_class

loc:@div*
T0*
_output_shapes
: 

gradients/lstm_1/mul_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/mul*
valueB *
_output_shapes
: 

!gradients/lstm_1/mul_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/mul*
valueB"      *
_output_shapes
:
ш
/gradients/lstm_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/lstm_1/mul_grad/Shape!gradients/lstm_1/mul_grad/Shape_1*
_class
loc:@lstm_1/mul*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/lstm_1/mul_grad/mulMulgradients/lstm_1/Sum_grad/Tilelstm_1/Square*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:
г
gradients/lstm_1/mul_grad/SumSumgradients/lstm_1/mul_grad/mul/gradients/lstm_1/mul_grad/BroadcastGradientArgs*
_class
loc:@lstm_1/mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
К
!gradients/lstm_1/mul_grad/ReshapeReshapegradients/lstm_1/mul_grad/Sumgradients/lstm_1/mul_grad/Shape*
Tshape0*
_class
loc:@lstm_1/mul*
T0*
_output_shapes
: 

gradients/lstm_1/mul_grad/mul_1Mullstm_1/mul/xgradients/lstm_1/Sum_grad/Tile*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:
й
gradients/lstm_1/mul_grad/Sum_1Sumgradients/lstm_1/mul_grad/mul_11gradients/lstm_1/mul_grad/BroadcastGradientArgs:1*
_class
loc:@lstm_1/mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ш
#gradients/lstm_1/mul_grad/Reshape_1Reshapegradients/lstm_1/mul_grad/Sum_1!gradients/lstm_1/mul_grad/Shape_1*
Tshape0*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:
Ж
$gradients/lstm_1/Square_1_grad/mul/xConst'^gradients/lstm_1/mul_11_grad/Reshape_1*
dtype0*"
_class
loc:@lstm_1/Square_1*
valueB
 *   @*
_output_shapes
: 
П
"gradients/lstm_1/Square_1_grad/mulMul$gradients/lstm_1/Square_1_grad/mul/xlstm_1/TensorArrayReadV3*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:џџџџџџџџџ
Э
$gradients/lstm_1/Square_1_grad/mul_1Mul&gradients/lstm_1/mul_11_grad/Reshape_1"gradients/lstm_1/Square_1_grad/mul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:џџџџџџџџџ
v
gradients/mul_grad/ShapeShapeMean_1*
_class

loc:@mul*
out_type0*
T0*
_output_shapes
:

gradients/mul_grad/Shape_1Shapedense_1_sample_weights*
_class

loc:@mul*
out_type0*
T0*
_output_shapes
:
Ь
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
_class

loc:@mul*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

gradients/mul_grad/mulMulgradients/div_grad/Reshapedense_1_sample_weights*
_class

loc:@mul*
T0*#
_output_shapes
:џџџџџџџџџ
З
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_class

loc:@mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ћ
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
Tshape0*
_class

loc:@mul*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/mul_grad/mul_1MulMean_1gradients/div_grad/Reshape*
_class

loc:@mul*
T0*#
_output_shapes
:џџџџџџџџџ
Н
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_class

loc:@mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Б
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
Tshape0*
_class

loc:@mul*
T0*#
_output_shapes
:џџџџџџџџџ
Џ
"gradients/lstm_1/Square_grad/mul/xConst$^gradients/lstm_1/mul_grad/Reshape_1*
dtype0* 
_class
loc:@lstm_1/Square*
valueB
 *   @*
_output_shapes
: 
Њ
 gradients/lstm_1/Square_grad/mulMul"gradients/lstm_1/Square_grad/mul/xlstm_1/kernel/read* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:
Л
"gradients/lstm_1/Square_grad/mul_1Mul#gradients/lstm_1/mul_grad/Reshape_1 gradients/lstm_1/Square_grad/mul* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:
z
gradients/Mean_1_grad/ShapeShapeMean*
_class
loc:@Mean_1*
out_type0*
T0*
_output_shapes
:
w
gradients/Mean_1_grad/SizeConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 

gradients/Mean_1_grad/addAddMean_1/reduction_indicesgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
: 

gradients/Mean_1_grad/modFloorModgradients/Mean_1_grad/addgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
: 

gradients/Mean_1_grad/Shape_1Const*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
~
!gradients/Mean_1_grad/range/startConst*
dtype0*
_class
loc:@Mean_1*
value	B : *
_output_shapes
: 
~
!gradients/Mean_1_grad/range/deltaConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
Щ
gradients/Mean_1_grad/rangeRange!gradients/Mean_1_grad/range/startgradients/Mean_1_grad/Size!gradients/Mean_1_grad/range/delta*
_class
loc:@Mean_1*

Tidx0*
_output_shapes
:
}
 gradients/Mean_1_grad/Fill/valueConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
Ё
gradients/Mean_1_grad/FillFillgradients/Mean_1_grad/Shape_1 gradients/Mean_1_grad/Fill/value*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
ї
#gradients/Mean_1_grad/DynamicStitchDynamicStitchgradients/Mean_1_grad/rangegradients/Mean_1_grad/modgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Fill*
N*
_class
loc:@Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
|
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
З
gradients/Mean_1_grad/MaximumMaximum#gradients/Mean_1_grad/DynamicStitchgradients/Mean_1_grad/Maximum/y*
_class
loc:@Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
Џ
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Maximum*
_class
loc:@Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
Е
gradients/Mean_1_grad/ReshapeReshapegradients/mul_grad/Reshape#gradients/Mean_1_grad/DynamicStitch*
Tshape0*
_class
loc:@Mean_1*
T0*
_output_shapes
:
Б
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/floordiv*

Tmultiples0*
_class
loc:@Mean_1*
T0*
_output_shapes
:
|
gradients/Mean_1_grad/Shape_2ShapeMean*
_class
loc:@Mean_1*
out_type0*
T0*
_output_shapes
:
~
gradients/Mean_1_grad/Shape_3ShapeMean_1*
_class
loc:@Mean_1*
out_type0*
T0*
_output_shapes
:

gradients/Mean_1_grad/ConstConst*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
З
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const*
_class
loc:@Mean_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 

gradients/Mean_1_grad/Const_1Const*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
Л
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_3gradients/Mean_1_grad/Const_1*
_class
loc:@Mean_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
~
!gradients/Mean_1_grad/Maximum_1/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
Ї
gradients/Mean_1_grad/Maximum_1Maximumgradients/Mean_1_grad/Prod_1!gradients/Mean_1_grad/Maximum_1/y*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
Ѕ
 gradients/Mean_1_grad/floordiv_1FloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum_1*
_class
loc:@Mean_1*
T0*
_output_shapes
: 

gradients/Mean_1_grad/CastCast gradients/Mean_1_grad/floordiv_1*

DstT0*
_class
loc:@Mean_1*

SrcT0*
_output_shapes
: 
Љ
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
_class
loc:@Mean_1*
T0*#
_output_shapes
:џџџџџџџџџ
x
gradients/Mean_grad/ShapeShapeSquare*
_class
	loc:@Mean*
out_type0*
T0*
_output_shapes
:
s
gradients/Mean_grad/SizeConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 

gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 

gradients/Mean_grad/modFloorModgradients/Mean_grad/addgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 
w
gradients/Mean_grad/Shape_1Const*
dtype0*
_class
	loc:@Mean*
valueB *
_output_shapes
: 
z
gradients/Mean_grad/range/startConst*
dtype0*
_class
	loc:@Mean*
value	B : *
_output_shapes
: 
z
gradients/Mean_grad/range/deltaConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
П
gradients/Mean_grad/rangeRangegradients/Mean_grad/range/startgradients/Mean_grad/Sizegradients/Mean_grad/range/delta*
_class
	loc:@Mean*

Tidx0*
_output_shapes
:
y
gradients/Mean_grad/Fill/valueConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 

gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
_class
	loc:@Mean*
T0*
_output_shapes
: 
ы
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
N*
_class
	loc:@Mean*
T0*#
_output_shapes
:џџџџџџџџџ
x
gradients/Mean_grad/Maximum/yConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
Џ
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
_class
	loc:@Mean*
T0*#
_output_shapes
:џџџџџџџџџ

gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*
T0*
_output_shapes
:
В
gradients/Mean_grad/ReshapeReshapegradients/Mean_1_grad/truediv!gradients/Mean_grad/DynamicStitch*
Tshape0*
_class
	loc:@Mean*
T0*
_output_shapes
:
С
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
_class
	loc:@Mean*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
z
gradients/Mean_grad/Shape_2ShapeSquare*
_class
	loc:@Mean*
out_type0*
T0*
_output_shapes
:
x
gradients/Mean_grad/Shape_3ShapeMean*
_class
	loc:@Mean*
out_type0*
T0*
_output_shapes
:
|
gradients/Mean_grad/ConstConst*
dtype0*
_class
	loc:@Mean*
valueB: *
_output_shapes
:
Џ
gradients/Mean_grad/ProdProdgradients/Mean_grad/Shape_2gradients/Mean_grad/Const*
_class
	loc:@Mean*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
~
gradients/Mean_grad/Const_1Const*
dtype0*
_class
	loc:@Mean*
valueB: *
_output_shapes
:
Г
gradients/Mean_grad/Prod_1Prodgradients/Mean_grad/Shape_3gradients/Mean_grad/Const_1*
_class
	loc:@Mean*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
z
gradients/Mean_grad/Maximum_1/yConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 

gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_class
	loc:@Mean*
T0*
_output_shapes
: 

gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
T0*
_output_shapes
: 

gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
_class
	loc:@Mean*

SrcT0*
_output_shapes
: 
Ў
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_class
	loc:@Mean*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ

gradients/Square_grad/mul/xConst^gradients/Mean_grad/truediv*
dtype0*
_class
loc:@Square*
valueB
 *   @*
_output_shapes
: 

gradients/Square_grad/mulMulgradients/Square_grad/mul/xsub*
_class
loc:@Square*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
А
gradients/Square_grad/mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/mul*
_class
loc:@Square*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
|
gradients/sub_grad/ShapeShapedense_1/Tanh*
_class

loc:@sub*
out_type0*
T0*
_output_shapes
:

gradients/sub_grad/Shape_1Shapedense_1_target*
_class

loc:@sub*
out_type0*
T0*
_output_shapes
:
Ь
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
_class

loc:@sub*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
М
gradients/sub_grad/SumSumgradients/Square_grad/mul_1(gradients/sub_grad/BroadcastGradientArgs*
_class

loc:@sub*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Џ
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*
_class

loc:@sub*
T0*'
_output_shapes
:џџџџџџџџџ
Р
gradients/sub_grad/Sum_1Sumgradients/Square_grad/mul_1*gradients/sub_grad/BroadcastGradientArgs:1*
_class

loc:@sub*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
r
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_class

loc:@sub*
T0*
_output_shapes
:
М
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*
_class

loc:@sub*
T0*0
_output_shapes
:џџџџџџџџџџџџџџџџџџ
­
$gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhgradients/sub_grad/Reshape*
_class
loc:@dense_1/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
П
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_1/Tanh_grad/TanhGrad*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
T0*
_output_shapes
:
ф
$gradients/dense_1/MatMul_grad/MatMulMatMul$gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
transpose_b(*
transpose_a( *!
_class
loc:@dense_1/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
т
&gradients/dense_1/MatMul_grad/MatMul_1MatMullstm_1/TensorArrayReadV3$gradients/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_1/MatMul*
T0*
_output_shapes

:
С
gradients/AddNAddN$gradients/lstm_1/Square_1_grad/mul_1$gradients/dense_1/MatMul_grad/MatMul*
N*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:џџџџџџџџџ
х
Igradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArraylstm_1/while/Exit_1*
source	gradients*%
_class
loc:@lstm_1/TensorArray*
_output_shapes

::
ќ
Egradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitylstm_1/while/Exit_1J^gradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
:
з
Kgradients/lstm_1/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Igradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3
lstm_1/subgradients/AddNEgradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
: 
h
gradients/zeros_like	ZerosLikelstm_1/while/Exit_2*
T0*'
_output_shapes
:џџџџџџџџџ
j
gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_3*
T0*'
_output_shapes
:џџџџџџџџџ
Ќ
)gradients/lstm_1/while/Exit_1_grad/b_exitEnterKgradients/lstm_1/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 

)gradients/lstm_1/while/Exit_2_grad/b_exitEntergradients/zeros_like*'
_output_shapes
:џџџџџџџџџ*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_2*
is_constant( 

)gradients/lstm_1/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*'
_output_shapes
:џџџџџџџџџ*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_3*
is_constant( 
ь
-gradients/lstm_1/while/Switch_1_grad/b_switchMerge)gradients/lstm_1/while/Exit_1_grad/b_exit4gradients/lstm_1/while/Switch_1_grad_1/NextIteration*
N*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: : 
§
-gradients/lstm_1/while/Switch_2_grad/b_switchMerge)gradients/lstm_1/while/Exit_2_grad/b_exit4gradients/lstm_1/while/Switch_2_grad_1/NextIteration*
N*'
_class
loc:@lstm_1/while/Merge_2*
T0*)
_output_shapes
:џџџџџџџџџ: 
§
-gradients/lstm_1/while/Switch_3_grad/b_switchMerge)gradients/lstm_1/while/Exit_3_grad/b_exit4gradients/lstm_1/while/Switch_3_grad_1/NextIteration*
N*'
_class
loc:@lstm_1/while/Merge_3*
T0*)
_output_shapes
:џџџџџџџџџ: 
Ф
*gradients/lstm_1/while/Merge_1_grad/SwitchSwitch-gradients/lstm_1/while/Switch_1_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: : 
ц
*gradients/lstm_1/while/Merge_2_grad/SwitchSwitch-gradients/lstm_1/while/Switch_2_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_2*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
ц
*gradients/lstm_1/while/Merge_3_grad/SwitchSwitch-gradients/lstm_1/while/Switch_3_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:џџџџџџџџџ:џџџџџџџџџ
І
(gradients/lstm_1/while/Enter_1_grad/ExitExit*gradients/lstm_1/while/Merge_1_grad/Switch*'
_class
loc:@lstm_1/while/Enter_1*
T0*
_output_shapes
: 
Д
ggradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/TensorArray*
is_constant(
ы
agradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3ggradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter,gradients/lstm_1/while/Merge_1_grad/Switch:1*
source	gradients*%
_class
loc:@lstm_1/TensorArray*
_output_shapes

::
У
]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity,gradients/lstm_1/while/Merge_1_grad/Switch:1b^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
: 
т
Wgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
	elem_type0*
_output_shapes
:

Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnterRefEnterWgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
is_constant(
ф
[gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPush	StackPushZgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnterlstm_1/while/Identity^gradients/Add*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
T0*
swap_memory(*
_output_shapes
:

cgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop/RefEnterRefEnterWgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
is_constant(
Ч
Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopStackPopcgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
	elem_type0*
_output_shapes
: 
Й+
Xgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger[^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopA^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_9_grad/mul/StackPop1^gradients/lstm_1/while/mul_9_grad/mul_1/StackPopB^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopD^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1K^gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopG^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopI^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1S^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1A^gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_4_grad/mul/StackPop1^gradients/lstm_1/while/mul_4_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_6_grad/mul/StackPop1^gradients/lstm_1/while/mul_6_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_8_grad/mul/StackPop1^gradients/lstm_1/while/mul_8_grad/mul_1/StackPopB^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopD^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1K^gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop@^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopB^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1I^gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1G^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopI^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1S^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopE^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopG^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1Q^gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1F^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopA^gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopF^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPopA^gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_7_grad/mul/StackPop1^gradients/lstm_1/while/mul_7_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_3_grad/mul/StackPop1^gradients/lstm_1/while/mul_3_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_1_grad/mul/StackPop1^gradients/lstm_1/while/mul_1_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_5_grad/mul/StackPopA^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1?^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1F^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPopD^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopF^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1F^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2F^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_35^gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPopA^gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_2_grad/mul/StackPop?^gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop-^gradients/lstm_1/while/mul_grad/mul/StackPop*%
_class
loc:@lstm_1/TensorArray
с
Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3agradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*%
_class
loc:@lstm_1/TensorArray*'
_output_shapes
:џџџџџџџџџ
§
gradients/AddN_1AddN,gradients/lstm_1/while/Merge_2_grad/Switch:1Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
N*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
'gradients/lstm_1/while/mul_9_grad/ShapeShapelstm_1/while/clip_by_value_2*%
_class
loc:@lstm_1/while/mul_9*
out_type0*
T0*
_output_shapes
:
Ѓ
)gradients/lstm_1/while/mul_9_grad/Shape_1Shapelstm_1/while/Tanh_1*%
_class
loc:@lstm_1/while/mul_9*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
І
Agradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_9_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
ћ
@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
Ќ
Cgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_9_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
џ
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_9*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Д
+gradients/lstm_1/while/mul_9_grad/mul/f_accStack*

stack_name *?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Ї
.gradients/lstm_1/while/mul_9_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_9_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
is_constant(

/gradients/lstm_1/while/mul_9_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_9_grad/mul/RefEnterlstm_1/while/Tanh_1^gradients/Add*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
К
7gradients/lstm_1/while/mul_9_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_9_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
is_constant(
ў
.gradients/lstm_1/while/mul_9_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_9_grad/mul/StackPop/RefEnter^gradients/Sub*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
Ч
%gradients/lstm_1/while/mul_9_grad/mulMulgradients/AddN_1.gradients/lstm_1/while/mul_9_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_9_grad/SumSum%gradients/lstm_1/while/mul_9_grad/mul7gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_9*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/mul_9_grad/ReshapeReshape%gradients/lstm_1/while/mul_9_grad/Sum@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ
П
-gradients/lstm_1/while/mul_9_grad/mul_1/f_accStack*

stack_name *H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
Д
0gradients/lstm_1/while/mul_9_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_9_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
is_constant(

1gradients/lstm_1/while/mul_9_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_9_grad/mul_1/RefEnterlstm_1/while/clip_by_value_2^gradients/Add*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
Ч
9gradients/lstm_1/while/mul_9_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_9_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
is_constant(

0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_9_grad/mul_1/StackPop/RefEnter^gradients/Sub*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
Ы
'gradients/lstm_1/while/mul_9_grad/mul_1Mul0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopgradients/AddN_1*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_9_grad/Sum_1Sum'gradients/lstm_1/while/mul_9_grad/mul_19gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_9*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/mul_9_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_9_grad/Sum_1Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
1gradients/lstm_1/while/clip_by_value_2_grad/ShapeShape$lstm_1/while/clip_by_value_2/Minimum*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
T0*
_output_shapes
:
З
3gradients/lstm_1/while/clip_by_value_2_grad/Shape_1Const^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB *
_output_shapes
: 
Э
3gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Shape)gradients/lstm_1/while/mul_9_grad/Reshape*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
T0*
_output_shapes
:
Н
7gradients/lstm_1/while/clip_by_value_2_grad/zeros/ConstConst^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
_output_shapes
: 

1gradients/lstm_1/while/clip_by_value_2_grad/zerosFill3gradients/lstm_1/while/clip_by_value_2_grad/Shape_27gradients/lstm_1/while/clip_by_value_2_grad/zeros/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ
т
>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_accStack*

stack_name *Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
ш
Agradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
к
Bgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush	StackPushAgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter$lstm_1/while/clip_by_value_2/Minimum^gradients/Add*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Jgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
П
Agradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopStackPopJgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
д
@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1Stack*

stack_name *J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
м
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1RefEnter@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
is_constant(
О
Dgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush_1	StackPushCgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1lstm_1/while/Const_4^gradients/Add*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
T0*
swap_memory(*
_output_shapes
:
я
Lgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1/RefEnterRefEnter@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
is_constant(
Ђ
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1StackPopLgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
: 
Г
8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqualGreaterEqualAgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopCgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ
Р
Ggradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
Я
Jgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
is_constant(
Ю
Kgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPush	StackPushJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnter1gradients/lstm_1/while/clip_by_value_2_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
swap_memory(*
_output_shapes
:
т
Sgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
is_constant(

Jgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopStackPopSgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
Щ
Agradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop3gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
2gradients/lstm_1/while/clip_by_value_2_grad/SelectSelect8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ
и
6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:џџџџџџџџџ
З
4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/lstm_1/while/clip_by_value_2_grad/SumSum2gradients/lstm_1/while/clip_by_value_2_grad/SelectAgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ќ
3gradients/lstm_1/while/clip_by_value_2_grad/ReshapeReshape/gradients/lstm_1/while/clip_by_value_2_grad/SumJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop*
Tshape0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:џџџџџџџџџ
Є
1gradients/lstm_1/while/clip_by_value_2_grad/Sum_1Sum4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Cgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

5gradients/lstm_1/while/clip_by_value_2_grad/Reshape_1Reshape1gradients/lstm_1/while/clip_by_value_2_grad/Sum_13gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*
Tshape0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
_output_shapes
: 
ю
+gradients/lstm_1/while/Tanh_1_grad/TanhGradTanhGrad.gradients/lstm_1/while/mul_9_grad/mul/StackPop+gradients/lstm_1/while/mul_9_grad/Reshape_1*&
_class
loc:@lstm_1/while/Tanh_1*
T0*'
_output_shapes
:џџџџџџџџџ
Н
4gradients/lstm_1/while/Switch_1_grad_1/NextIterationNextIteration,gradients/lstm_1/while/Merge_1_grad/Switch:1*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: 
Ф
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ShapeShapelstm_1/while/add_7*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
T0*
_output_shapes
:
Ч
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB *
_output_shapes
: 
ч
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2Shape3gradients/lstm_1/while/clip_by_value_2_grad/Reshape*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
T0*
_output_shapes
:
Э
?gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *    *
_output_shapes
: 
Њ
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosFill;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2?gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
н
Cgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_accStack*

stack_name *P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
ш
Fgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
Ш
Ggradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush	StackPushFgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_7^gradients/Add*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Ogradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
П
Fgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopStackPopOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
с
Egradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
ю
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1RefEnterEgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
а
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush_1	StackPushHgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_5^gradients/Add*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:

Qgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
Д
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1StackPopQgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
: 
Ч
=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual	LessEqualFgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopHgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
а
Ogradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
ч
Rgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
ю
Sgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnter9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape^gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
њ
[gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
Б
Rgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopStackPop[gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
щ
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
:gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectSelect=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
э
>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:џџџџџџџџџ
с
<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
О
7gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumSum:gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectIgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ь
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeReshape7gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop*
Tshape0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1Sum<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Kgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs:1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ј
=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape_1Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*
Tshape0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
_output_shapes
: 
з
gradients/AddN_2AddN,gradients/lstm_1/while/Merge_3_grad/Switch:1+gradients/lstm_1/while/Tanh_1_grad/TanhGrad*
N*'
_class
loc:@lstm_1/while/Merge_3*
T0*'
_output_shapes
:џџџџџџџџџ
 
'gradients/lstm_1/while/add_5_grad/ShapeShapelstm_1/while/mul_4*%
_class
loc:@lstm_1/while/add_5*
out_type0*
T0*
_output_shapes
:
Ђ
)gradients/lstm_1/while/add_5_grad/Shape_1Shapelstm_1/while/mul_6*%
_class
loc:@lstm_1/while/add_5*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
І
Agradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_5_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_5*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
ћ
@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
Ќ
Cgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_5_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_5*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
џ
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_5*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
%gradients/lstm_1/while/add_5_grad/SumSumgradients/AddN_27gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/add_5_grad/ReshapeReshape%gradients/lstm_1/while/add_5_grad/Sum@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
т
'gradients/lstm_1/while/add_5_grad/Sum_1Sumgradients/AddN_29gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/add_5_grad/Reshape_1Reshape'gradients/lstm_1/while/add_5_grad/Sum_1Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/add_5*
T0*'
_output_shapes
:џџџџџџџџџ
 
'gradients/lstm_1/while/add_7_grad/ShapeShapelstm_1/while/mul_8*%
_class
loc:@lstm_1/while/add_7*
out_type0*
T0*
_output_shapes
:
Ѓ
)gradients/lstm_1/while/add_7_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_7*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_7*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_7*
is_constant(
І
Agradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_7_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_7*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_7*
is_constant(
ћ
@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_7*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/add_7_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_7_grad/Shape_1*%
_class
loc:@lstm_1/while/add_7*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

%gradients/lstm_1/while/add_7_grad/SumSum;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape7gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/add_7_grad/ReshapeReshape%gradients/lstm_1/while/add_7_grad/Sum@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_7*
T0*'
_output_shapes
:џџџџџџџџџ

'gradients/lstm_1/while/add_7_grad/Sum_1Sum;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape9gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
р
+gradients/lstm_1/while/add_7_grad/Reshape_1Reshape'gradients/lstm_1/while/add_7_grad/Sum_1)gradients/lstm_1/while/add_7_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/add_7*
T0*
_output_shapes
: 
Њ
'gradients/lstm_1/while/mul_4_grad/ShapeShapelstm_1/while/clip_by_value_1*%
_class
loc:@lstm_1/while/mul_4*
out_type0*
T0*
_output_shapes
:
Ї
)gradients/lstm_1/while/mul_4_grad/Shape_1Shapelstm_1/while/Identity_3*%
_class
loc:@lstm_1/while/mul_4*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
І
Agradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_4_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
ћ
@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
Ќ
Cgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_4_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
џ
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_4*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
И
+gradients/lstm_1/while/mul_4_grad/mul/f_accStack*

stack_name *C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Ћ
.gradients/lstm_1/while/mul_4_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_4_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
is_constant(

/gradients/lstm_1/while/mul_4_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_4_grad/mul/RefEnterlstm_1/while/Identity_3^gradients/Add*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
О
7gradients/lstm_1/while/mul_4_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_4_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
is_constant(

.gradients/lstm_1/while/mul_4_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_4_grad/mul/StackPop/RefEnter^gradients/Sub*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_4_grad/mulMul)gradients/lstm_1/while/add_5_grad/Reshape.gradients/lstm_1/while/mul_4_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_4_grad/SumSum%gradients/lstm_1/while/mul_4_grad/mul7gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/mul_4_grad/ReshapeReshape%gradients/lstm_1/while/mul_4_grad/Sum@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:џџџџџџџџџ
П
-gradients/lstm_1/while/mul_4_grad/mul_1/f_accStack*

stack_name *H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
Д
0gradients/lstm_1/while/mul_4_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_4_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
is_constant(

1gradients/lstm_1/while/mul_4_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_4_grad/mul_1/RefEnterlstm_1/while/clip_by_value_1^gradients/Add*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
Ч
9gradients/lstm_1/while/mul_4_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_4_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
is_constant(

0gradients/lstm_1/while/mul_4_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_4_grad/mul_1/StackPop/RefEnter^gradients/Sub*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
ф
'gradients/lstm_1/while/mul_4_grad/mul_1Mul0gradients/lstm_1/while/mul_4_grad/mul_1/StackPop)gradients/lstm_1/while/add_5_grad/Reshape*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_4_grad/Sum_1Sum'gradients/lstm_1/while/mul_4_grad/mul_19gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/mul_4_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_4_grad/Sum_1Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:џџџџџџџџџ
Ј
'gradients/lstm_1/while/mul_6_grad/ShapeShapelstm_1/while/clip_by_value*%
_class
loc:@lstm_1/while/mul_6*
out_type0*
T0*
_output_shapes
:
Ё
)gradients/lstm_1/while/mul_6_grad/Shape_1Shapelstm_1/while/Tanh*%
_class
loc:@lstm_1/while/mul_6*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
І
Agradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_6_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
ћ
@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
Ќ
Cgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_6_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
џ
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_6*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
В
+gradients/lstm_1/while/mul_6_grad/mul/f_accStack*

stack_name *=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
Ѕ
.gradients/lstm_1/while/mul_6_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_6_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
is_constant(

/gradients/lstm_1/while/mul_6_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_6_grad/mul/RefEnterlstm_1/while/Tanh^gradients/Add*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
И
7gradients/lstm_1/while/mul_6_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_6_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
is_constant(
ќ
.gradients/lstm_1/while/mul_6_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_6_grad/mul/StackPop/RefEnter^gradients/Sub*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
т
%gradients/lstm_1/while/mul_6_grad/mulMul+gradients/lstm_1/while/add_5_grad/Reshape_1.gradients/lstm_1/while/mul_6_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_6_grad/SumSum%gradients/lstm_1/while/mul_6_grad/mul7gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_6*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/mul_6_grad/ReshapeReshape%gradients/lstm_1/while/mul_6_grad/Sum@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ
Н
-gradients/lstm_1/while/mul_6_grad/mul_1/f_accStack*

stack_name *F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
В
0gradients/lstm_1/while/mul_6_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_6_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
is_constant(

1gradients/lstm_1/while/mul_6_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_6_grad/mul_1/RefEnterlstm_1/while/clip_by_value^gradients/Add*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
Х
9gradients/lstm_1/while/mul_6_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_6_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
is_constant(

0gradients/lstm_1/while/mul_6_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_6_grad/mul_1/StackPop/RefEnter^gradients/Sub*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
ц
'gradients/lstm_1/while/mul_6_grad/mul_1Mul0gradients/lstm_1/while/mul_6_grad/mul_1/StackPop+gradients/lstm_1/while/add_5_grad/Reshape_1*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_6_grad/Sum_1Sum'gradients/lstm_1/while/mul_6_grad/mul_19gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_6*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/mul_6_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_6_grad/Sum_1Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
'gradients/lstm_1/while/mul_8_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_8*
valueB *
_output_shapes
: 
Ђ
)gradients/lstm_1/while/mul_8_grad/Shape_1Shapelstm_1/while/add_6*%
_class
loc:@lstm_1/while/mul_8*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_8*
is_constant(
Ј
Agradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_8_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_8*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_8*
is_constant(
ћ
@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:

7gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_8_grad/Shape@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Г
+gradients/lstm_1/while/mul_8_grad/mul/f_accStack*

stack_name *>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
І
.gradients/lstm_1/while/mul_8_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_8_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
is_constant(

/gradients/lstm_1/while/mul_8_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_8_grad/mul/RefEnterlstm_1/while/add_6^gradients/Add*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
T0*
swap_memory(*
_output_shapes
:
Й
7gradients/lstm_1/while/mul_8_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_8_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
is_constant(
§
.gradients/lstm_1/while/mul_8_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_8_grad/mul/StackPop/RefEnter^gradients/Sub*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_8_grad/mulMul)gradients/lstm_1/while/add_7_grad/Reshape.gradients/lstm_1/while/mul_8_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_8_grad/SumSum%gradients/lstm_1/while/mul_8_grad/mul7gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_8*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
к
)gradients/lstm_1/while/mul_8_grad/ReshapeReshape%gradients/lstm_1/while/mul_8_grad/Sum'gradients/lstm_1/while/mul_8_grad/Shape*
Tshape0*%
_class
loc:@lstm_1/while/mul_8*
T0*
_output_shapes
: 
З
-gradients/lstm_1/while/mul_8_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
	elem_type0*
_output_shapes
:
Ќ
0gradients/lstm_1/while/mul_8_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_8_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
is_constant(

1gradients/lstm_1/while/mul_8_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_8_grad/mul_1/RefEnterlstm_1/while/mul_8/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
T0*
swap_memory(*
_output_shapes
:
П
9gradients/lstm_1/while/mul_8_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_8_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
is_constant(
ђ
0gradients/lstm_1/while/mul_8_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_8_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
	elem_type0*
_output_shapes
: 
ф
'gradients/lstm_1/while/mul_8_grad/mul_1Mul0gradients/lstm_1/while/mul_8_grad/mul_1/StackPop)gradients/lstm_1/while/add_7_grad/Reshape*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_8_grad/Sum_1Sum'gradients/lstm_1/while/mul_8_grad/mul_19gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_8*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/mul_8_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_8_grad/Sum_1@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:џџџџџџџџџ
Ц
1gradients/lstm_1/while/clip_by_value_1_grad/ShapeShape$lstm_1/while/clip_by_value_1/Minimum*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
T0*
_output_shapes
:
З
3gradients/lstm_1/while/clip_by_value_1_grad/Shape_1Const^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB *
_output_shapes
: 
Э
3gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Shape)gradients/lstm_1/while/mul_4_grad/Reshape*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
T0*
_output_shapes
:
Н
7gradients/lstm_1/while/clip_by_value_1_grad/zeros/ConstConst^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB
 *    *
_output_shapes
: 

1gradients/lstm_1/while/clip_by_value_1_grad/zerosFill3gradients/lstm_1/while/clip_by_value_1_grad/Shape_27gradients/lstm_1/while/clip_by_value_1_grad/zeros/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ
т
>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_accStack*

stack_name *Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
ш
Agradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
к
Bgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush	StackPushAgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter$lstm_1/while/clip_by_value_1/Minimum^gradients/Add*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Jgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
П
Agradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopStackPopJgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
д
@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1Stack*

stack_name *J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
м
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1RefEnter@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
is_constant(
О
Dgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush_1	StackPushCgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1lstm_1/while/Const_2^gradients/Add*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
T0*
swap_memory(*
_output_shapes
:
я
Lgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1/RefEnterRefEnter@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
is_constant(
Ђ
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1StackPopLgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
: 
Г
8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqualGreaterEqualAgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopCgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ
Р
Ggradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
Я
Jgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
is_constant(
Ю
Kgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPush	StackPushJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnter1gradients/lstm_1/while/clip_by_value_1_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
swap_memory(*
_output_shapes
:
т
Sgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
is_constant(

Jgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopStackPopSgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
Щ
Agradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop3gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
З
2gradients/lstm_1/while/clip_by_value_1_grad/SelectSelect8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ
и
6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:џџџџџџџџџ
З
4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/lstm_1/while/clip_by_value_1_grad/SumSum2gradients/lstm_1/while/clip_by_value_1_grad/SelectAgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ќ
3gradients/lstm_1/while/clip_by_value_1_grad/ReshapeReshape/gradients/lstm_1/while/clip_by_value_1_grad/SumJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop*
Tshape0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:џџџџџџџџџ
Є
1gradients/lstm_1/while/clip_by_value_1_grad/Sum_1Sum4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Cgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

5gradients/lstm_1/while/clip_by_value_1_grad/Reshape_1Reshape1gradients/lstm_1/while/clip_by_value_1_grad/Sum_13gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*
Tshape0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
_output_shapes
: 
Р
/gradients/lstm_1/while/clip_by_value_grad/ShapeShape"lstm_1/while/clip_by_value/Minimum*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
T0*
_output_shapes
:
Г
1gradients/lstm_1/while/clip_by_value_grad/Shape_1Const^gradients/Sub*
dtype0*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB *
_output_shapes
: 
Щ
1gradients/lstm_1/while/clip_by_value_grad/Shape_2Shape)gradients/lstm_1/while/mul_6_grad/Reshape*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
T0*
_output_shapes
:
Й
5gradients/lstm_1/while/clip_by_value_grad/zeros/ConstConst^gradients/Sub*
dtype0*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    *
_output_shapes
: 

/gradients/lstm_1/while/clip_by_value_grad/zerosFill1gradients/lstm_1/while/clip_by_value_grad/Shape_25gradients/lstm_1/while/clip_by_value_grad/zeros/Const*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
м
<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_accStack*

stack_name *V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
р
?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnterRefEnter<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
а
@gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush	StackPush?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter"lstm_1/while/clip_by_value/Minimum^gradients/Add*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
ѓ
Hgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop/RefEnterRefEnter<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
З
?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopStackPopHgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
Ю
>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1Stack*

stack_name *F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
д
Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1RefEnter>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
is_constant(
Д
Bgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush_1	StackPushAgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1lstm_1/while/Const^gradients/Add*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
T0*
swap_memory(*
_output_shapes
:
ч
Jgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
is_constant(

Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1StackPopJgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
: 
Ћ
6gradients/lstm_1/while/clip_by_value_grad/GreaterEqualGreaterEqual?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopAgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
М
Egradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_accStack*

stack_name *-
_class#
!loc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
Щ
Hgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
is_constant(
Ц
Igradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPush	StackPushHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnter/gradients/lstm_1/while/clip_by_value_grad/Shape^gradients/Add*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
swap_memory(*
_output_shapes
:
м
Qgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
is_constant(

Hgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopStackPopQgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
С
?gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop1gradients/lstm_1/while/clip_by_value_grad/Shape_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Џ
0gradients/lstm_1/while/clip_by_value_grad/SelectSelect6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ
в
4gradients/lstm_1/while/clip_by_value_grad/LogicalNot
LogicalNot6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:џџџџџџџџџ
Џ
2gradients/lstm_1/while/clip_by_value_grad/Select_1Select4gradients/lstm_1/while/clip_by_value_grad/LogicalNot)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ

-gradients/lstm_1/while/clip_by_value_grad/SumSum0gradients/lstm_1/while/clip_by_value_grad/Select?gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs*-
_class#
!loc:@lstm_1/while/clip_by_value*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Є
1gradients/lstm_1/while/clip_by_value_grad/ReshapeReshape-gradients/lstm_1/while/clip_by_value_grad/SumHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop*
Tshape0*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:џџџџџџџџџ

/gradients/lstm_1/while/clip_by_value_grad/Sum_1Sum2gradients/lstm_1/while/clip_by_value_grad/Select_1Agradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs:1*-
_class#
!loc:@lstm_1/while/clip_by_value*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

3gradients/lstm_1/while/clip_by_value_grad/Reshape_1Reshape/gradients/lstm_1/while/clip_by_value_grad/Sum_11gradients/lstm_1/while/clip_by_value_grad/Shape_1*
Tshape0*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
_output_shapes
: 
ъ
)gradients/lstm_1/while/Tanh_grad/TanhGradTanhGrad.gradients/lstm_1/while/mul_6_grad/mul/StackPop+gradients/lstm_1/while/mul_6_grad/Reshape_1*$
_class
loc:@lstm_1/while/Tanh*
T0*'
_output_shapes
:џџџџџџџџџ
Њ
'gradients/lstm_1/while/add_6_grad/ShapeShapelstm_1/while/strided_slice_3*%
_class
loc:@lstm_1/while/add_6*
out_type0*
T0*
_output_shapes
:
Ѕ
)gradients/lstm_1/while/add_6_grad/Shape_1Shapelstm_1/while/MatMul_3*%
_class
loc:@lstm_1/while/add_6*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
І
Agradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_6_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_6*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
ћ
@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
Ќ
Cgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_6_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_6*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
џ
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/add_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_6*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
%gradients/lstm_1/while/add_6_grad/SumSum+gradients/lstm_1/while/mul_8_grad/Reshape_17gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_6*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/add_6_grad/ReshapeReshape%gradients/lstm_1/while/add_6_grad/Sum@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
§
'gradients/lstm_1/while/add_6_grad/Sum_1Sum+gradients/lstm_1/while/mul_8_grad/Reshape_19gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_6*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/add_6_grad/Reshape_1Reshape'gradients/lstm_1/while/add_6_grad/Sum_1Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/add_6*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ShapeShapelstm_1/while/add_3*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0*
T0*
_output_shapes
:
Ч
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB *
_output_shapes
: 
ч
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2Shape3gradients/lstm_1/while/clip_by_value_1_grad/Reshape*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0*
T0*
_output_shapes
:
Э
?gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB
 *    *
_output_shapes
: 
Њ
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosFill;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2?gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
н
Cgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_accStack*

stack_name *P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
ш
Fgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
Ш
Ggradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush	StackPushFgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_3^gradients/Add*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Ogradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
П
Fgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopStackPopOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
с
Egradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
ю
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1RefEnterEgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
а
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush_1	StackPushHgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_3^gradients/Add*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:

Qgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
Д
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1StackPopQgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
: 
Ч
=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual	LessEqualFgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopHgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
а
Ogradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
ч
Rgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
ю
Sgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnter9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape^gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
њ
[gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
Б
Rgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopStackPop[gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
щ
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
о
:gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectSelect=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
э
>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:џџџџџџџџџ
с
<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
О
7gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumSum:gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectIgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ь
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeReshape7gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop*
Tshape0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
Ф
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1Sum<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Kgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ј
=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape_1Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*
Tshape0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
_output_shapes
: 
Э
4gradients/lstm_1/while/Switch_3_grad_1/NextIterationNextIteration+gradients/lstm_1/while/mul_4_grad/Reshape_1*'
_class
loc:@lstm_1/while/Merge_3*
T0*'
_output_shapes
:џџџџџџџџџ
Р
7gradients/lstm_1/while/clip_by_value/Minimum_grad/ShapeShapelstm_1/while/add_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
T0*
_output_shapes
:
У
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB *
_output_shapes
: 
с
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Shape1gradients/lstm_1/while/clip_by_value_grad/Reshape*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
T0*
_output_shapes
:
Щ
=gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *    *
_output_shapes
: 
Ђ
7gradients/lstm_1/while/clip_by_value/Minimum_grad/zerosFill9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2=gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/Const*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
й
Agradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_accStack*

stack_name *N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
т
Dgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnterRefEnterAgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Т
Egradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush	StackPushDgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_1^gradients/Add*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
ѕ
Mgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterAgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Й
Dgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopStackPopMgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
н
Cgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
ш
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1RefEnterCgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Ъ
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush_1	StackPushFgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_1^gradients/Add*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
ћ
Ogradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Ў
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1StackPopOgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
: 
П
;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual	LessEqualDgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopFgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
Ь
Mgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
с
Pgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterMgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
ц
Qgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnter7gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape^gradients/Add*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
є
Ygradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterMgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
Ћ
Pgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopStackPopYgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
с
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
д
8gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectSelect;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
ч
<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot
LogicalNot;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:џџџџџџџџџ
з
:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Select<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
Ж
5gradients/lstm_1/while/clip_by_value/Minimum_grad/SumSum8gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectGgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
Ф
9gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeReshape5gradients/lstm_1/while/clip_by_value/Minimum_grad/SumPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop*
Tshape0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:џџџџџџџџџ
М
7gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_1Sum:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Igradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
 
;gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape_1Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_19gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*
Tshape0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
_output_shapes
: 
Њ
'gradients/lstm_1/while/add_4_grad/ShapeShapelstm_1/while/strided_slice_2*%
_class
loc:@lstm_1/while/add_4*
out_type0*
T0*
_output_shapes
:
Ѕ
)gradients/lstm_1/while/add_4_grad/Shape_1Shapelstm_1/while/MatMul_2*%
_class
loc:@lstm_1/while/add_4*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
І
Agradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_4_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_4*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
ћ
@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
Ќ
Cgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_4_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_4*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
џ
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_4*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ї
%gradients/lstm_1/while/add_4_grad/SumSum)gradients/lstm_1/while/Tanh_grad/TanhGrad7gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/add_4_grad/ReshapeReshape%gradients/lstm_1/while/add_4_grad/Sum@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_4*
T0*'
_output_shapes
:џџџџџџџџџ
ћ
'gradients/lstm_1/while/add_4_grad/Sum_1Sum)gradients/lstm_1/while/Tanh_grad/TanhGrad9gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/add_4_grad/Reshape_1Reshape'gradients/lstm_1/while/add_4_grad/Sum_1Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/add_4*
T0*'
_output_shapes
:џџџџџџџџџ
Р
1gradients/lstm_1/while/strided_slice_3_grad/ShapeShapelstm_1/while/TensorArrayReadV3*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
out_type0*
T0*
_output_shapes
:
Л
Bgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_3*
	elem_type0*
_output_shapes
:
Х
Egradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
is_constant(
Ф
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_3_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*
swap_memory(*
_output_shapes
:
и
Ngradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
is_constant(

Egradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
	elem_type0*
_output_shapes
:
ц
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
	elem_type0*
_output_shapes
:
ђ
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
is_constant(
т
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_3/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
is_constant(
М
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_3/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_3/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
	elem_type0*
_output_shapes
:
ё
<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_6_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_3

1gradients/lstm_1/while/MatMul_3_grad/MatMul/EnterEnterlstm_1/strided_slice_7*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_3*
is_constant(

+gradients/lstm_1/while/MatMul_3_grad/MatMulMatMul+gradients/lstm_1/while/add_6_grad/Reshape_11gradients/lstm_1/while/MatMul_3_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_3*
T0*'
_output_shapes
:џџџџџџџџџ
О
3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
Й
6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
is_constant(

7gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnterlstm_1/while/mul_7^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
Ь
?gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
is_constant(

6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
	elem_type0*'
_output_shapes
:џџџџџџџџџ

-gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_6_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_3*
T0*
_output_shapes

:
 
'gradients/lstm_1/while/add_3_grad/ShapeShapelstm_1/while/mul_3*%
_class
loc:@lstm_1/while/add_3*
out_type0*
T0*
_output_shapes
:
Ѓ
)gradients/lstm_1/while/add_3_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_3*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_3*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_3*
is_constant(
І
Agradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_3_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_3*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_3*
is_constant(
ћ
@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_3*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_3_grad/Shape_1*%
_class
loc:@lstm_1/while/add_3*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

%gradients/lstm_1/while/add_3_grad/SumSum;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape7gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/add_3_grad/ReshapeReshape%gradients/lstm_1/while/add_3_grad/Sum@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_3*
T0*'
_output_shapes
:џџџџџџџџџ

'gradients/lstm_1/while/add_3_grad/Sum_1Sum;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape9gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
р
+gradients/lstm_1/while/add_3_grad/Reshape_1Reshape'gradients/lstm_1/while/add_3_grad/Sum_1)gradients/lstm_1/while/add_3_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/add_3*
T0*
_output_shapes
: 
 
'gradients/lstm_1/while/add_1_grad/ShapeShapelstm_1/while/mul_1*%
_class
loc:@lstm_1/while/add_1*
out_type0*
T0*
_output_shapes
:
Ѓ
)gradients/lstm_1/while/add_1_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_1*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_1*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_1*
is_constant(
І
Agradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_1_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_1*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_1*
is_constant(
ћ
@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_1_grad/Shape_1*%
_class
loc:@lstm_1/while/add_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ

%gradients/lstm_1/while/add_1_grad/SumSum9gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape7gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/add_1_grad/ReshapeReshape%gradients/lstm_1/while/add_1_grad/Sum@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_1*
T0*'
_output_shapes
:џџџџџџџџџ

'gradients/lstm_1/while/add_1_grad/Sum_1Sum9gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape9gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
р
+gradients/lstm_1/while/add_1_grad/Reshape_1Reshape'gradients/lstm_1/while/add_1_grad/Sum_1)gradients/lstm_1/while/add_1_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/add_1*
T0*
_output_shapes
: 
Р
1gradients/lstm_1/while/strided_slice_2_grad/ShapeShapelstm_1/while/TensorArrayReadV3*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
out_type0*
T0*
_output_shapes
:
Л
Bgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_2*
	elem_type0*
_output_shapes
:
Х
Egradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
is_constant(
Ф
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_2_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
T0*
swap_memory(*
_output_shapes
:
и
Ngradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
is_constant(

Egradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
	elem_type0*
_output_shapes
:
ц
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
	elem_type0*
_output_shapes
:
ђ
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
is_constant(
т
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_2/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
is_constant(
М
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_2/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_2/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
	elem_type0*
_output_shapes
:
ё
<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_4_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_2

1gradients/lstm_1/while/MatMul_2_grad/MatMul/EnterEnterlstm_1/strided_slice_6*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_2*
is_constant(

+gradients/lstm_1/while/MatMul_2_grad/MatMulMatMul+gradients/lstm_1/while/add_4_grad/Reshape_11gradients/lstm_1/while/MatMul_2_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_2*
T0*'
_output_shapes
:џџџџџџџџџ
О
3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
Й
6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
is_constant(

7gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnterlstm_1/while/mul_5^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
T0*
swap_memory(*
_output_shapes
:
Ь
?gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
is_constant(

6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
	elem_type0*'
_output_shapes
:џџџџџџџџџ

-gradients/lstm_1/while/MatMul_2_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_4_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_2*
T0*
_output_shapes

:
Ѕ
'gradients/lstm_1/while/mul_7_grad/ShapeShapelstm_1/while/Identity_2*%
_class
loc:@lstm_1/while/mul_7*
out_type0*
T0*
_output_shapes
:
Ѓ
)gradients/lstm_1/while/mul_7_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_7*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_7*
is_constant(
І
Agradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_7_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_7*
is_constant(
ћ
@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_7_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_7*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Е
+gradients/lstm_1/while/mul_7_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
	elem_type0*
_output_shapes
:
Ј
.gradients/lstm_1/while/mul_7_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_7_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
is_constant(

/gradients/lstm_1/while/mul_7_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_7_grad/mul/RefEnterlstm_1/while/mul_7/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
T0*
swap_memory(*
_output_shapes
:
Л
7gradients/lstm_1/while/mul_7_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_7_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
is_constant(
ю
.gradients/lstm_1/while/mul_7_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_7_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
	elem_type0*
_output_shapes
: 
т
%gradients/lstm_1/while/mul_7_grad/mulMul+gradients/lstm_1/while/MatMul_3_grad/MatMul.gradients/lstm_1/while/mul_7_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_7_grad/SumSum%gradients/lstm_1/while/mul_7_grad/mul7gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/mul_7_grad/ReshapeReshape%gradients/lstm_1/while/mul_7_grad/Sum@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
К
-gradients/lstm_1/while/mul_7_grad/mul_1/f_accStack*

stack_name *C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
Џ
0gradients/lstm_1/while/mul_7_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_7_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
is_constant(

1gradients/lstm_1/while/mul_7_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_7_grad/mul_1/RefEnterlstm_1/while/Identity_2^gradients/Add*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
Т
9gradients/lstm_1/while/mul_7_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_7_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
is_constant(

0gradients/lstm_1/while/mul_7_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_7_grad/mul_1/StackPop/RefEnter^gradients/Sub*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
ц
'gradients/lstm_1/while/mul_7_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_3_grad/MatMul*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_7_grad/Sum_1Sum'gradients/lstm_1/while/mul_7_grad/mul_19gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
р
+gradients/lstm_1/while/mul_7_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_7_grad/Sum_1)gradients/lstm_1/while/mul_7_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_7*
T0*
_output_shapes
: 
Е
0gradients/lstm_1/while/MatMul_3/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
valueB*    *
_output_shapes

:
Њ
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
is_constant( 

2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_3/Enter_grad/NextIteration*
N*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0* 
_output_shapes
:: 
ч
1gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*(
_output_shapes
::
ђ
.gradients/lstm_1/while/MatMul_3/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_3_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
в
8gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_3/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
Ц
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
Ё
'gradients/lstm_1/while/mul_3_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_3*
valueB *
_output_shapes
: 
Ђ
)gradients/lstm_1/while/mul_3_grad/Shape_1Shapelstm_1/while/add_2*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_3*
is_constant(
Ј
Agradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_3_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_3*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_3*
is_constant(
ћ
@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:

7gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_3_grad/Shape@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Г
+gradients/lstm_1/while/mul_3_grad/mul/f_accStack*

stack_name *>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
І
.gradients/lstm_1/while/mul_3_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_3_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
is_constant(

/gradients/lstm_1/while/mul_3_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_3_grad/mul/RefEnterlstm_1/while/add_2^gradients/Add*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
T0*
swap_memory(*
_output_shapes
:
Й
7gradients/lstm_1/while/mul_3_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_3_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
is_constant(
§
.gradients/lstm_1/while/mul_3_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_3_grad/mul/StackPop/RefEnter^gradients/Sub*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_3_grad/mulMul)gradients/lstm_1/while/add_3_grad/Reshape.gradients/lstm_1/while/mul_3_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_3_grad/SumSum%gradients/lstm_1/while/mul_3_grad/mul7gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
к
)gradients/lstm_1/while/mul_3_grad/ReshapeReshape%gradients/lstm_1/while/mul_3_grad/Sum'gradients/lstm_1/while/mul_3_grad/Shape*
Tshape0*%
_class
loc:@lstm_1/while/mul_3*
T0*
_output_shapes
: 
З
-gradients/lstm_1/while/mul_3_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
	elem_type0*
_output_shapes
:
Ќ
0gradients/lstm_1/while/mul_3_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_3_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
is_constant(

1gradients/lstm_1/while/mul_3_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_3_grad/mul_1/RefEnterlstm_1/while/mul_3/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
T0*
swap_memory(*
_output_shapes
:
П
9gradients/lstm_1/while/mul_3_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_3_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
is_constant(
ђ
0gradients/lstm_1/while/mul_3_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_3_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
	elem_type0*
_output_shapes
: 
ф
'gradients/lstm_1/while/mul_3_grad/mul_1Mul0gradients/lstm_1/while/mul_3_grad/mul_1/StackPop)gradients/lstm_1/while/add_3_grad/Reshape*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_3_grad/Sum_1Sum'gradients/lstm_1/while/mul_3_grad/mul_19gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/mul_3_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_3_grad/Sum_1@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:џџџџџџџџџ
Ё
'gradients/lstm_1/while/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_1*
valueB *
_output_shapes
: 
 
)gradients/lstm_1/while/mul_1_grad/Shape_1Shapelstm_1/while/add*%
_class
loc:@lstm_1/while/mul_1*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_1*
is_constant(
Ј
Agradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_1_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_1*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_1*
is_constant(
ћ
@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:

7gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_1_grad/Shape@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Б
+gradients/lstm_1/while/mul_1_grad/mul/f_accStack*

stack_name *<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
Є
.gradients/lstm_1/while/mul_1_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_1_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
is_constant(

/gradients/lstm_1/while/mul_1_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_1_grad/mul/RefEnterlstm_1/while/add^gradients/Add*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
T0*
swap_memory(*
_output_shapes
:
З
7gradients/lstm_1/while/mul_1_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_1_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
is_constant(
ћ
.gradients/lstm_1/while/mul_1_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
	elem_type0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_1_grad/mulMul)gradients/lstm_1/while/add_1_grad/Reshape.gradients/lstm_1/while/mul_1_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_1_grad/SumSum%gradients/lstm_1/while/mul_1_grad/mul7gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
к
)gradients/lstm_1/while/mul_1_grad/ReshapeReshape%gradients/lstm_1/while/mul_1_grad/Sum'gradients/lstm_1/while/mul_1_grad/Shape*
Tshape0*%
_class
loc:@lstm_1/while/mul_1*
T0*
_output_shapes
: 
З
-gradients/lstm_1/while/mul_1_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
	elem_type0*
_output_shapes
:
Ќ
0gradients/lstm_1/while/mul_1_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_1_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
is_constant(

1gradients/lstm_1/while/mul_1_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_1_grad/mul_1/RefEnterlstm_1/while/mul_1/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
T0*
swap_memory(*
_output_shapes
:
П
9gradients/lstm_1/while/mul_1_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_1_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
is_constant(
ђ
0gradients/lstm_1/while/mul_1_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
	elem_type0*
_output_shapes
: 
ф
'gradients/lstm_1/while/mul_1_grad/mul_1Mul0gradients/lstm_1/while/mul_1_grad/mul_1/StackPop)gradients/lstm_1/while/add_1_grad/Reshape*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_1_grad/Sum_1Sum'gradients/lstm_1/while/mul_1_grad/mul_19gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/mul_1_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_1_grad/Sum_1@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:џџџџџџџџџ
Ѕ
'gradients/lstm_1/while/mul_5_grad/ShapeShapelstm_1/while/Identity_2*%
_class
loc:@lstm_1/while/mul_5*
out_type0*
T0*
_output_shapes
:
Ѓ
)gradients/lstm_1/while/mul_5_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_5*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_5*
is_constant(
І
Agradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_5_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_5*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_5*
is_constant(
ћ
@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_5_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_5*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Е
+gradients/lstm_1/while/mul_5_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
	elem_type0*
_output_shapes
:
Ј
.gradients/lstm_1/while/mul_5_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_5_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
is_constant(

/gradients/lstm_1/while/mul_5_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_5_grad/mul/RefEnterlstm_1/while/mul_5/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
T0*
swap_memory(*
_output_shapes
:
Л
7gradients/lstm_1/while/mul_5_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_5_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
is_constant(
ю
.gradients/lstm_1/while/mul_5_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_5_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
	elem_type0*
_output_shapes
: 
т
%gradients/lstm_1/while/mul_5_grad/mulMul+gradients/lstm_1/while/MatMul_2_grad/MatMul.gradients/lstm_1/while/mul_5_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_5_grad/SumSum%gradients/lstm_1/while/mul_5_grad/mul7gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/mul_5_grad/ReshapeReshape%gradients/lstm_1/while/mul_5_grad/Sum@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:џџџџџџџџџ
ц
'gradients/lstm_1/while/mul_5_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_2_grad/MatMul*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_5_grad/Sum_1Sum'gradients/lstm_1/while/mul_5_grad/mul_19gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
р
+gradients/lstm_1/while/mul_5_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_5_grad/Sum_1)gradients/lstm_1/while/mul_5_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_5*
T0*
_output_shapes
: 
Е
0gradients/lstm_1/while/MatMul_2/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
valueB*    *
_output_shapes

:
Њ
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
is_constant( 

2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_2/Enter_grad/NextIteration*
N*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0* 
_output_shapes
:: 
ч
1gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*(
_output_shapes
::
ђ
.gradients/lstm_1/while/MatMul_2/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_2_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
в
8gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_2/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
Ц
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
Ї
+gradients/lstm_1/strided_slice_7_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_7*
valueB"      *
_output_shapes
:
Ю
6gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_22gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_7
Њ
'gradients/lstm_1/while/add_2_grad/ShapeShapelstm_1/while/strided_slice_1*%
_class
loc:@lstm_1/while/add_2*
out_type0*
T0*
_output_shapes
:
Ѕ
)gradients/lstm_1/while/add_2_grad/Shape_1Shapelstm_1/while/MatMul_1*%
_class
loc:@lstm_1/while/add_2*
out_type0*
T0*
_output_shapes
:
Ќ
=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
І
Agradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_2_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_2*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
ћ
@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
Ў
?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
Е
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
Ќ
Cgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_2_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_2*
T0*
swap_memory(*
_output_shapes
:
Ш
Kgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
џ
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
К
7gradients/lstm_1/while/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_2*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
љ
%gradients/lstm_1/while/add_2_grad/SumSum+gradients/lstm_1/while/mul_3_grad/Reshape_17gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/add_2_grad/ReshapeReshape%gradients/lstm_1/while/add_2_grad/Sum@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_2*
T0*'
_output_shapes
:џџџџџџџџџ
§
'gradients/lstm_1/while/add_2_grad/Sum_1Sum+gradients/lstm_1/while/mul_3_grad/Reshape_19gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

+gradients/lstm_1/while/add_2_grad/Reshape_1Reshape'gradients/lstm_1/while/add_2_grad/Sum_1Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/add_2*
T0*'
_output_shapes
:џџџџџџџџџ
Є
%gradients/lstm_1/while/add_grad/ShapeShapelstm_1/while/strided_slice*#
_class
loc:@lstm_1/while/add*
out_type0*
T0*
_output_shapes
:

'gradients/lstm_1/while/add_grad/Shape_1Shapelstm_1/while/MatMul*#
_class
loc:@lstm_1/while/add*
out_type0*
T0*
_output_shapes
:
Ј
;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_accStack*

stack_name *#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
Ћ
>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnterRefEnter;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(

?gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush	StackPush>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter%gradients/lstm_1/while/add_grad/Shape^gradients/Add*#
_class
loc:@lstm_1/while/add*
T0*
swap_memory(*
_output_shapes
:
О
Ggradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
ѕ
>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPopStackPopGgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
Њ
=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
Џ
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1RefEnter=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
Є
Agradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush_1	StackPush@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1'gradients/lstm_1/while/add_grad/Shape_1^gradients/Add*#
_class
loc:@lstm_1/while/add*
T0*
swap_memory(*
_output_shapes
:
Т
Igradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
љ
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1StackPopIgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
В
5gradients/lstm_1/while/add_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1*#
_class
loc:@lstm_1/while/add*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
ѓ
#gradients/lstm_1/while/add_grad/SumSum+gradients/lstm_1/while/mul_1_grad/Reshape_15gradients/lstm_1/while/add_grad/BroadcastGradientArgs*#
_class
loc:@lstm_1/while/add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
ќ
'gradients/lstm_1/while/add_grad/ReshapeReshape#gradients/lstm_1/while/add_grad/Sum>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop*
Tshape0*#
_class
loc:@lstm_1/while/add*
T0*'
_output_shapes
:џџџџџџџџџ
ї
%gradients/lstm_1/while/add_grad/Sum_1Sum+gradients/lstm_1/while/mul_1_grad/Reshape_17gradients/lstm_1/while/add_grad/BroadcastGradientArgs:1*#
_class
loc:@lstm_1/while/add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/add_grad/Reshape_1Reshape%gradients/lstm_1/while/add_grad/Sum_1@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*#
_class
loc:@lstm_1/while/add*
T0*'
_output_shapes
:џџџџџџџџџ
Ї
+gradients/lstm_1/strided_slice_6_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_6*
valueB"      *
_output_shapes
:
Ю
6gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_22gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_6
Р
1gradients/lstm_1/while/strided_slice_1_grad/ShapeShapelstm_1/while/TensorArrayReadV3*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
out_type0*
T0*
_output_shapes
:
Л
Bgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_1*
	elem_type0*
_output_shapes
:
Х
Egradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
is_constant(
Ф
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_1_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
T0*
swap_memory(*
_output_shapes
:
и
Ngradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
is_constant(

Egradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
	elem_type0*
_output_shapes
:
ц
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
	elem_type0*
_output_shapes
:
ђ
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
is_constant(
т
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_1/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
is_constant(
М
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_1/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
	elem_type0*
_output_shapes
:
ш
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
	elem_type0*
_output_shapes
:
є
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
is_constant(
ц
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_1/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
T0*
swap_memory(*
_output_shapes
:

Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
is_constant(
О
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
	elem_type0*
_output_shapes
:
ё
<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_2_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_1

1gradients/lstm_1/while/MatMul_1_grad/MatMul/EnterEnterlstm_1/strided_slice_5*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_1*
is_constant(

+gradients/lstm_1/while/MatMul_1_grad/MatMulMatMul+gradients/lstm_1/while/add_2_grad/Reshape_11gradients/lstm_1/while/MatMul_1_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ
О
3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
Й
6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
is_constant(

7gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnterlstm_1/while/mul_2^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
T0*
swap_memory(*
_output_shapes
:
Ь
?gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
is_constant(

6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
	elem_type0*'
_output_shapes
:џџџџџџџџџ

-gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_2_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_1*
T0*
_output_shapes

:
М
/gradients/lstm_1/while/strided_slice_grad/ShapeShapelstm_1/while/TensorArrayReadV3*-
_class#
!loc:@lstm_1/while/strided_slice*
out_type0*
T0*
_output_shapes
:
З
@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_accStack*

stack_name *-
_class#
!loc:@lstm_1/while/strided_slice*
	elem_type0*
_output_shapes
:
П
Cgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnterRefEnter@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/strided_slice*
is_constant(
М
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush	StackPushCgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter/gradients/lstm_1/while/strided_slice_grad/Shape^gradients/Add*-
_class#
!loc:@lstm_1/while/strided_slice*
T0*
swap_memory(*
_output_shapes
:
в
Lgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop/RefEnterRefEnter@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/strided_slice*
is_constant(

Cgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopStackPopLgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*-
_class#
!loc:@lstm_1/while/strided_slice*
	elem_type0*
_output_shapes
:
р
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
	elem_type0*
_output_shapes
:
ъ
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
is_constant(
и
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_1	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1 lstm_1/while/strided_slice/stack^gradients/Add*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
T0*
swap_memory(*
_output_shapes
:
§
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
is_constant(
Д
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
	elem_type0*
_output_shapes
:
т
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
	elem_type0*
_output_shapes
:
ь
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
is_constant(
м
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_2	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2"lstm_1/while/strided_slice/stack_1^gradients/Add*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
T0*
swap_memory(*
_output_shapes
:
џ
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
is_constant(
Ж
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
	elem_type0*
_output_shapes
:
т
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
	elem_type0*
_output_shapes
:
ь
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
is_constant(
м
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_3	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3"lstm_1/while/strided_slice/stack_2^gradients/Add*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
T0*
swap_memory(*
_output_shapes
:
џ
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
is_constant(
Ж
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
	elem_type0*
_output_shapes
:
у
:gradients/lstm_1/while/strided_slice_grad/StridedSliceGradStridedSliceGradCgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3'gradients/lstm_1/while/add_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:џџџџџџџџџ*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *-
_class#
!loc:@lstm_1/while/strided_slice

/gradients/lstm_1/while/MatMul_grad/MatMul/EnterEnterlstm_1/strided_slice_4*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/MatMul*
is_constant(

)gradients/lstm_1/while/MatMul_grad/MatMulMatMul)gradients/lstm_1/while/add_grad/Reshape_1/gradients/lstm_1/while/MatMul_grad/MatMul/Enter*
transpose_b(*
transpose_a( *&
_class
loc:@lstm_1/while/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
И
1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_accStack*

stack_name *=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
Б
4gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnterRefEnter1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
is_constant(

5gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPush	StackPush4gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnterlstm_1/while/mul^gradients/Add*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
T0*
swap_memory(*
_output_shapes
:
Ф
=gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnter1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
is_constant(

4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPopStackPop=gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
	elem_type0*'
_output_shapes
:џџџџџџџџџ

+gradients/lstm_1/while/MatMul_grad/MatMul_1MatMul4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop)gradients/lstm_1/while/add_grad/Reshape_1*
transpose_b( *
transpose_a(*&
_class
loc:@lstm_1/while/MatMul*
T0*
_output_shapes

:
Ѕ
'gradients/lstm_1/while/mul_2_grad/ShapeShapelstm_1/while/Identity_2*%
_class
loc:@lstm_1/while/mul_2*
out_type0*
T0*
_output_shapes
:
Ѓ
)gradients/lstm_1/while/mul_2_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_2*
valueB *
_output_shapes
: 
Ќ
=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
Б
@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_2*
is_constant(
І
Agradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_2_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_2*
T0*
swap_memory(*
_output_shapes
:
Ф
Igradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_2*
is_constant(
ћ
@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
Ё
7gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_2_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_2*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Е
+gradients/lstm_1/while/mul_2_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
	elem_type0*
_output_shapes
:
Ј
.gradients/lstm_1/while/mul_2_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_2_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
is_constant(

/gradients/lstm_1/while/mul_2_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_2_grad/mul/RefEnterlstm_1/while/mul_2/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
T0*
swap_memory(*
_output_shapes
:
Л
7gradients/lstm_1/while/mul_2_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_2_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
is_constant(
ю
.gradients/lstm_1/while/mul_2_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
	elem_type0*
_output_shapes
: 
т
%gradients/lstm_1/while/mul_2_grad/mulMul+gradients/lstm_1/while/MatMul_1_grad/MatMul.gradients/lstm_1/while/mul_2_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
ѓ
%gradients/lstm_1/while/mul_2_grad/SumSum%gradients/lstm_1/while/mul_2_grad/mul7gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:

)gradients/lstm_1/while/mul_2_grad/ReshapeReshape%gradients/lstm_1/while/mul_2_grad/Sum@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
ц
'gradients/lstm_1/while/mul_2_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_1_grad/MatMul*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:џџџџџџџџџ
љ
'gradients/lstm_1/while/mul_2_grad/Sum_1Sum'gradients/lstm_1/while/mul_2_grad/mul_19gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
р
+gradients/lstm_1/while/mul_2_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_2_grad/Sum_1)gradients/lstm_1/while/mul_2_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_2*
T0*
_output_shapes
: 
Е
0gradients/lstm_1/while/MatMul_1/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
valueB*    *
_output_shapes

:
Њ
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
is_constant( 

2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_1/Enter_grad/NextIteration*
N*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0* 
_output_shapes
:: 
ч
1gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*(
_output_shapes
::
ђ
.gradients/lstm_1/while/MatMul_1/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_1_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
в
8gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_1/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
Ц
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
њ
gradients/AddN_3AddN<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad:gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad*
N*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*'
_output_shapes
:џџџџџџџџџ
І
Ugradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
б
Wgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(

Ogradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Ugradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterWgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
source	gradients*'
_class
loc:@lstm_1/TensorArray_1*
_output_shapes

::
Ь
Kgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityWgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1P^gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
Н
Qgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ogradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopgradients/AddN_3Kgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
Ё
%gradients/lstm_1/while/mul_grad/ShapeShapelstm_1/while/Identity_2*#
_class
loc:@lstm_1/while/mul*
out_type0*
T0*
_output_shapes
:

'gradients/lstm_1/while/mul_grad/Shape_1Const^gradients/Sub*
dtype0*#
_class
loc:@lstm_1/while/mul*
valueB *
_output_shapes
: 
Ј
;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_accStack*

stack_name *#
_class
loc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
Ћ
>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnterRefEnter;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/mul*
is_constant(

?gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPush	StackPush>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnter%gradients/lstm_1/while/mul_grad/Shape^gradients/Add*#
_class
loc:@lstm_1/while/mul*
T0*
swap_memory(*
_output_shapes
:
О
Ggradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/mul*
is_constant(
ѕ
>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopStackPopGgradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:

5gradients/lstm_1/while/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop'gradients/lstm_1/while/mul_grad/Shape_1*#
_class
loc:@lstm_1/while/mul*
T0*2
_output_shapes 
:џџџџџџџџџ:џџџџџџџџџ
Џ
)gradients/lstm_1/while/mul_grad/mul/f_accStack*

stack_name *<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
	elem_type0*
_output_shapes
:
 
,gradients/lstm_1/while/mul_grad/mul/RefEnterRefEnter)gradients/lstm_1/while/mul_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
is_constant(

-gradients/lstm_1/while/mul_grad/mul/StackPush	StackPush,gradients/lstm_1/while/mul_grad/mul/RefEnterlstm_1/while/mul/y^gradients/Add*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
T0*
swap_memory(*
_output_shapes
:
Г
5gradients/lstm_1/while/mul_grad/mul/StackPop/RefEnterRefEnter)gradients/lstm_1/while/mul_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
is_constant(
ц
,gradients/lstm_1/while/mul_grad/mul/StackPopStackPop5gradients/lstm_1/while/mul_grad/mul/StackPop/RefEnter^gradients/Sub*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
	elem_type0*
_output_shapes
: 
к
#gradients/lstm_1/while/mul_grad/mulMul)gradients/lstm_1/while/MatMul_grad/MatMul,gradients/lstm_1/while/mul_grad/mul/StackPop*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:џџџџџџџџџ
ы
#gradients/lstm_1/while/mul_grad/SumSum#gradients/lstm_1/while/mul_grad/mul5gradients/lstm_1/while/mul_grad/BroadcastGradientArgs*#
_class
loc:@lstm_1/while/mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
ќ
'gradients/lstm_1/while/mul_grad/ReshapeReshape#gradients/lstm_1/while/mul_grad/Sum>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop*
Tshape0*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:џџџџџџџџџ
р
%gradients/lstm_1/while/mul_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop)gradients/lstm_1/while/MatMul_grad/MatMul*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:џџџџџџџџџ
ё
%gradients/lstm_1/while/mul_grad/Sum_1Sum%gradients/lstm_1/while/mul_grad/mul_17gradients/lstm_1/while/mul_grad/BroadcastGradientArgs:1*#
_class
loc:@lstm_1/while/mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
и
)gradients/lstm_1/while/mul_grad/Reshape_1Reshape%gradients/lstm_1/while/mul_grad/Sum_1'gradients/lstm_1/while/mul_grad/Shape_1*
Tshape0*#
_class
loc:@lstm_1/while/mul*
T0*
_output_shapes
: 
Б
.gradients/lstm_1/while/MatMul/Enter_grad/b_accConst*
dtype0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
valueB*    *
_output_shapes

:
Є
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter.gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
is_constant( 

0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2Merge0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_16gradients/lstm_1/while/MatMul/Enter_grad/NextIteration*
N*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0* 
_output_shapes
:: 
с
/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*(
_output_shapes
::
ъ
,gradients/lstm_1/while/MatMul/Enter_grad/AddAdd1gradients/lstm_1/while/MatMul/Enter_grad/Switch:1+gradients/lstm_1/while/MatMul_grad/MatMul_1*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
Ь
6gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration,gradients/lstm_1/while/MatMul/Enter_grad/Add*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
Р
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit/gradients/lstm_1/while/MatMul/Enter_grad/Switch*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
Ї
+gradients/lstm_1/strided_slice_5_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_5*
valueB"      *
_output_shapes
:
Ю
6gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_22gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_5
Љ
;gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*
valueB
 *    *
_output_shapes
: 
Б
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter;gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant( 

=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Cgradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: : 
ц
<gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: : 

9gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/AddAdd>gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch:1Qgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
й
Cgradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration9gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Add*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
Э
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit<gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
Є
gradients/AddN_4AddN)gradients/lstm_1/while/mul_7_grad/Reshape)gradients/lstm_1/while/mul_5_grad/Reshape)gradients/lstm_1/while/mul_2_grad/Reshape'gradients/lstm_1/while/mul_grad/Reshape*
N*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:џџџџџџџџџ
Ї
+gradients/lstm_1/strided_slice_4_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_4*
valueB"      *
_output_shapes
:
Ь
6gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_20gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_4
М
rgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArray_1=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*'
_class
loc:@lstm_1/TensorArray_1*
_output_shapes

::
ј
ngradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3s^gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
њ
dgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3rgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lstm_1/TensorArrayUnstack/rangengradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*+
_output_shapes
:(џџџџџџџџџ
В
4gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_4*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:џџџџџџџџџ
е
gradients/AddN_5AddN6gradients/lstm_1/strided_slice_7_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_6_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_5_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*
N*)
_class
loc:@lstm_1/strided_slice_7*
T0*
_output_shapes

:
Ї
1gradients/lstm_1/transpose_grad/InvertPermutationInvertPermutationlstm_1/transpose/perm*#
_class
loc:@lstm_1/transpose*
T0*
_output_shapes
:
З
)gradients/lstm_1/transpose_grad/transpose	Transposedgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV31gradients/lstm_1/transpose_grad/InvertPermutation*#
_class
loc:@lstm_1/transpose*
Tperm0*
T0*+
_output_shapes
:џџџџџџџџџ(

!gradients/lstm_1/concat_grad/RankConst*
dtype0* 
_class
loc:@lstm_1/concat*
value	B :*
_output_shapes
: 
І
 gradients/lstm_1/concat_grad/modFloorModlstm_1/concat/axis!gradients/lstm_1/concat_grad/Rank* 
_class
loc:@lstm_1/concat*
T0*
_output_shapes
: 

"gradients/lstm_1/concat_grad/ShapeShapelstm_1/Reshape_1* 
_class
loc:@lstm_1/concat*
out_type0*
T0*
_output_shapes
:
ч
#gradients/lstm_1/concat_grad/ShapeNShapeNlstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7*
N*
out_type0* 
_class
loc:@lstm_1/concat*,
_output_shapes
::::*
T0
Ю
)gradients/lstm_1/concat_grad/ConcatOffsetConcatOffset gradients/lstm_1/concat_grad/mod#gradients/lstm_1/concat_grad/ShapeN%gradients/lstm_1/concat_grad/ShapeN:1%gradients/lstm_1/concat_grad/ShapeN:2%gradients/lstm_1/concat_grad/ShapeN:3*
N* 
_class
loc:@lstm_1/concat*,
_output_shapes
::::

"gradients/lstm_1/concat_grad/SliceSlice)gradients/lstm_1/transpose_grad/transpose)gradients/lstm_1/concat_grad/ConcatOffset#gradients/lstm_1/concat_grad/ShapeN*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:џџџџџџџџџ(

$gradients/lstm_1/concat_grad/Slice_1Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:1%gradients/lstm_1/concat_grad/ShapeN:1*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:џџџџџџџџџ(

$gradients/lstm_1/concat_grad/Slice_2Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:2%gradients/lstm_1/concat_grad/ShapeN:2*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:џџџџџџџџџ(

$gradients/lstm_1/concat_grad/Slice_3Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:3%gradients/lstm_1/concat_grad/ShapeN:3*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:џџџџџџџџџ(

%gradients/lstm_1/Reshape_1_grad/ShapeShapelstm_1/BiasAdd*#
_class
loc:@lstm_1/Reshape_1*
out_type0*
T0*
_output_shapes
:
т
'gradients/lstm_1/Reshape_1_grad/ReshapeReshape"gradients/lstm_1/concat_grad/Slice%gradients/lstm_1/Reshape_1_grad/Shape*
Tshape0*#
_class
loc:@lstm_1/Reshape_1*
T0*'
_output_shapes
:џџџџџџџџџ

%gradients/lstm_1/Reshape_3_grad/ShapeShapelstm_1/BiasAdd_1*#
_class
loc:@lstm_1/Reshape_3*
out_type0*
T0*
_output_shapes
:
ф
'gradients/lstm_1/Reshape_3_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_1%gradients/lstm_1/Reshape_3_grad/Shape*
Tshape0*#
_class
loc:@lstm_1/Reshape_3*
T0*'
_output_shapes
:џџџџџџџџџ

%gradients/lstm_1/Reshape_5_grad/ShapeShapelstm_1/BiasAdd_2*#
_class
loc:@lstm_1/Reshape_5*
out_type0*
T0*
_output_shapes
:
ф
'gradients/lstm_1/Reshape_5_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_2%gradients/lstm_1/Reshape_5_grad/Shape*
Tshape0*#
_class
loc:@lstm_1/Reshape_5*
T0*'
_output_shapes
:џџџџџџџџџ

%gradients/lstm_1/Reshape_7_grad/ShapeShapelstm_1/BiasAdd_3*#
_class
loc:@lstm_1/Reshape_7*
out_type0*
T0*
_output_shapes
:
ф
'gradients/lstm_1/Reshape_7_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_3%gradients/lstm_1/Reshape_7_grad/Shape*
Tshape0*#
_class
loc:@lstm_1/Reshape_7*
T0*'
_output_shapes
:џџџџџџџџџ
Р
)gradients/lstm_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_1_grad/Reshape*!
_class
loc:@lstm_1/BiasAdd*
data_formatNHWC*
T0*
_output_shapes
:
Ф
+gradients/lstm_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_3_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_1*
data_formatNHWC*
T0*
_output_shapes
:
Ф
+gradients/lstm_1/BiasAdd_2_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_5_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_2*
data_formatNHWC*
T0*
_output_shapes
:
Ф
+gradients/lstm_1/BiasAdd_3_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_7_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_3*
data_formatNHWC*
T0*
_output_shapes
:
ц
#gradients/lstm_1/MatMul_grad/MatMulMatMul'gradients/lstm_1/Reshape_1_grad/Reshapelstm_1/strided_slice*
transpose_b(*
transpose_a( * 
_class
loc:@lstm_1/MatMul*
T0*'
_output_shapes
:џџџџџџџџџ
й
%gradients/lstm_1/MatMul_grad/MatMul_1MatMullstm_1/Reshape'gradients/lstm_1/Reshape_1_grad/Reshape*
transpose_b( *
transpose_a(* 
_class
loc:@lstm_1/MatMul*
T0*
_output_shapes

:
 
+gradients/lstm_1/strided_slice_8_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_8*
valueB:*
_output_shapes
:
С
6gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2)gradients/lstm_1/BiasAdd_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_8
ь
%gradients/lstm_1/MatMul_1_grad/MatMulMatMul'gradients/lstm_1/Reshape_3_grad/Reshapelstm_1/strided_slice_1*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_1*
T0*'
_output_shapes
:џџџџџџџџџ
п
'gradients/lstm_1/MatMul_1_grad/MatMul_1MatMullstm_1/Reshape_2'gradients/lstm_1/Reshape_3_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_1*
T0*
_output_shapes

:
 
+gradients/lstm_1/strided_slice_9_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_9*
valueB:*
_output_shapes
:
У
6gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2+gradients/lstm_1/BiasAdd_1_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_9
ь
%gradients/lstm_1/MatMul_2_grad/MatMulMatMul'gradients/lstm_1/Reshape_5_grad/Reshapelstm_1/strided_slice_2*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_2*
T0*'
_output_shapes
:џџџџџџџџџ
п
'gradients/lstm_1/MatMul_2_grad/MatMul_1MatMullstm_1/Reshape_4'gradients/lstm_1/Reshape_5_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_2*
T0*
_output_shapes

:
Ђ
,gradients/lstm_1/strided_slice_10_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_10*
valueB:*
_output_shapes
:
Щ
7gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2+gradients/lstm_1/BiasAdd_2_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask **
_class 
loc:@lstm_1/strided_slice_10
ь
%gradients/lstm_1/MatMul_3_grad/MatMulMatMul'gradients/lstm_1/Reshape_7_grad/Reshapelstm_1/strided_slice_3*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_3*
T0*'
_output_shapes
:џџџџџџџџџ
п
'gradients/lstm_1/MatMul_3_grad/MatMul_1MatMullstm_1/Reshape_6'gradients/lstm_1/Reshape_7_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_3*
T0*
_output_shapes

:
Ђ
,gradients/lstm_1/strided_slice_11_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_11*
valueB:*
_output_shapes
:
Щ
7gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2+gradients/lstm_1/BiasAdd_3_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask **
_class 
loc:@lstm_1/strided_slice_11
Ѓ
)gradients/lstm_1/strided_slice_grad/ShapeConst*
dtype0*'
_class
loc:@lstm_1/strided_slice*
valueB"      *
_output_shapes
:
Е
4gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad)gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2%gradients/lstm_1/MatMul_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *'
_class
loc:@lstm_1/strided_slice
Ї
+gradients/lstm_1/strided_slice_1_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_1*
valueB"      *
_output_shapes
:
У
6gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2'gradients/lstm_1/MatMul_1_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_1
Ї
+gradients/lstm_1/strided_slice_2_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_2*
valueB"      *
_output_shapes
:
У
6gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2'gradients/lstm_1/MatMul_2_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_2
Ї
+gradients/lstm_1/strided_slice_3_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_3*
valueB"      *
_output_shapes
:
У
6gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2'gradients/lstm_1/MatMul_3_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_3
г
gradients/AddN_6AddN6gradients/lstm_1/strided_slice_8_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_9_grad/StridedSliceGrad7gradients/lstm_1/strided_slice_10_grad/StridedSliceGrad7gradients/lstm_1/strided_slice_11_grad/StridedSliceGrad*
N*)
_class
loc:@lstm_1/strided_slice_8*
T0*
_output_shapes
:
ю
gradients/AddN_7AddN"gradients/lstm_1/Square_grad/mul_14gradients/lstm_1/strided_slice_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_1_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_2_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_3_grad/StridedSliceGrad*
N* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:
T
Const_2Const*
dtype0*
valueB*    *
_output_shapes
:
t
Variable
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:

Variable/AssignAssignVariableConst_2*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
:
e
Variable/readIdentityVariable*
_class
loc:@Variable*
T0*
_output_shapes
:
\
Const_3Const*
dtype0*
valueB*    *
_output_shapes

:
~

Variable_1
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
Ё
Variable_1/AssignAssign
Variable_1Const_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes

:
T
Const_4Const*
dtype0*
valueB*    *
_output_shapes
:
v

Variable_2
VariableV2*
dtype0*
shape:*
shared_name *
	container *
_output_shapes
:

Variable_2/AssignAssign
Variable_2Const_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:
k
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes
:
\
Const_5Const*
dtype0*
valueB*    *
_output_shapes

:
~

Variable_3
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
Ё
Variable_3/AssignAssign
Variable_3Const_5*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes

:
o
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes

:
\
Const_6Const*
dtype0*
valueB*    *
_output_shapes

:
~

Variable_4
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
Ё
Variable_4/AssignAssign
Variable_4Const_6*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:
o
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes

:
J
mul_2Mulrho/readVariable/read*
T0*
_output_shapes
:
L
sub_1/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_1Subsub_1/xrho/read*
T0*
_output_shapes
: 
c
Square_1Square*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
B
mul_3Mulsub_1Square_1*
T0*
_output_shapes
:
?
add_2Addmul_2mul_3*
T0*
_output_shapes
:

AssignAssignVariableadd_2*
validate_shape(*
_class
loc:@Variable*
use_locking(*
T0*
_output_shapes
:
f
mul_4Mullr/read*gradients/dense_1/BiasAdd_grad/BiasAddGrad*
T0*
_output_shapes
:
L
Const_7Const*
dtype0*
valueB
 *    *
_output_shapes
: 
L
Const_8Const*
dtype0*
valueB
 *  *
_output_shapes
: 
U
clip_by_value/MinimumMinimumadd_2Const_8*
T0*
_output_shapes
:
]
clip_by_valueMaximumclip_by_value/MinimumConst_7*
T0*
_output_shapes
:
@
SqrtSqrtclip_by_value*
T0*
_output_shapes
:
L
add_3/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
@
add_3AddSqrtadd_3/y*
T0*
_output_shapes
:
C
div_1RealDivmul_4add_3*
T0*
_output_shapes
:
K
sub_2Subdense_1/bias/readdiv_1*
T0*
_output_shapes
:

Assign_1Assigndense_1/biassub_2*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0*
_output_shapes
:
P
mul_5Mulrho/readVariable_1/read*
T0*
_output_shapes

:
L
sub_3/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_3Subsub_3/xrho/read*
T0*
_output_shapes
: 
c
Square_2Square&gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
F
mul_6Mulsub_3Square_2*
T0*
_output_shapes

:
C
add_4Addmul_5mul_6*
T0*
_output_shapes

:

Assign_2Assign
Variable_1add_4*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:
f
mul_7Mullr/read&gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
L
Const_9Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_10Const*
dtype0*
valueB
 *  *
_output_shapes
: 
\
clip_by_value_1/MinimumMinimumadd_4Const_10*
T0*
_output_shapes

:
e
clip_by_value_1Maximumclip_by_value_1/MinimumConst_9*
T0*
_output_shapes

:
H
Sqrt_1Sqrtclip_by_value_1*
T0*
_output_shapes

:
L
add_5/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
F
add_5AddSqrt_1add_5/y*
T0*
_output_shapes

:
G
div_2RealDivmul_7add_5*
T0*
_output_shapes

:
Q
sub_4Subdense_1/kernel/readdiv_2*
T0*
_output_shapes

:

Assign_3Assigndense_1/kernelsub_4*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:
L
mul_8Mulrho/readVariable_2/read*
T0*
_output_shapes
:
L
sub_5/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_5Subsub_5/xrho/read*
T0*
_output_shapes
: 
I
Square_3Squaregradients/AddN_6*
T0*
_output_shapes
:
B
mul_9Mulsub_5Square_3*
T0*
_output_shapes
:
?
add_6Addmul_8mul_9*
T0*
_output_shapes
:

Assign_4Assign
Variable_2add_6*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:
M
mul_10Mullr/readgradients/AddN_6*
T0*
_output_shapes
:
M
Const_11Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_12Const*
dtype0*
valueB
 *  *
_output_shapes
: 
X
clip_by_value_2/MinimumMinimumadd_6Const_12*
T0*
_output_shapes
:
b
clip_by_value_2Maximumclip_by_value_2/MinimumConst_11*
T0*
_output_shapes
:
D
Sqrt_2Sqrtclip_by_value_2*
T0*
_output_shapes
:
L
add_7/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
B
add_7AddSqrt_2add_7/y*
T0*
_output_shapes
:
D
div_3RealDivmul_10add_7*
T0*
_output_shapes
:
J
sub_6Sublstm_1/bias/readdiv_3*
T0*
_output_shapes
:

Assign_5Assignlstm_1/biassub_6*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes
:
Q
mul_11Mulrho/readVariable_3/read*
T0*
_output_shapes

:
L
sub_7/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_7Subsub_7/xrho/read*
T0*
_output_shapes
: 
M
Square_4Squaregradients/AddN_7*
T0*
_output_shapes

:
G
mul_12Mulsub_7Square_4*
T0*
_output_shapes

:
E
add_8Addmul_11mul_12*
T0*
_output_shapes

:

Assign_6Assign
Variable_3add_8*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes

:
Q
mul_13Mullr/readgradients/AddN_7*
T0*
_output_shapes

:
M
Const_13Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_14Const*
dtype0*
valueB
 *  *
_output_shapes
: 
\
clip_by_value_3/MinimumMinimumadd_8Const_14*
T0*
_output_shapes

:
f
clip_by_value_3Maximumclip_by_value_3/MinimumConst_13*
T0*
_output_shapes

:
H
Sqrt_3Sqrtclip_by_value_3*
T0*
_output_shapes

:
L
add_9/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
F
add_9AddSqrt_3add_9/y*
T0*
_output_shapes

:
H
div_4RealDivmul_13add_9*
T0*
_output_shapes

:
P
sub_8Sublstm_1/kernel/readdiv_4*
T0*
_output_shapes

:

Assign_7Assignlstm_1/kernelsub_8*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes

:
Q
mul_14Mulrho/readVariable_4/read*
T0*
_output_shapes

:
L
sub_9/xConst*
dtype0*
valueB
 *  ?*
_output_shapes
: 
@
sub_9Subsub_9/xrho/read*
T0*
_output_shapes
: 
M
Square_5Squaregradients/AddN_5*
T0*
_output_shapes

:
G
mul_15Mulsub_9Square_5*
T0*
_output_shapes

:
F
add_10Addmul_14mul_15*
T0*
_output_shapes

:

Assign_8Assign
Variable_4add_10*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:
Q
mul_16Mullr/readgradients/AddN_5*
T0*
_output_shapes

:
M
Const_15Const*
dtype0*
valueB
 *    *
_output_shapes
: 
M
Const_16Const*
dtype0*
valueB
 *  *
_output_shapes
: 
]
clip_by_value_4/MinimumMinimumadd_10Const_16*
T0*
_output_shapes

:
f
clip_by_value_4Maximumclip_by_value_4/MinimumConst_15*
T0*
_output_shapes

:
H
Sqrt_4Sqrtclip_by_value_4*
T0*
_output_shapes

:
M
add_11/yConst*
dtype0*
valueB
 *wЬ+2*
_output_shapes
: 
H
add_11AddSqrt_4add_11/y*
T0*
_output_shapes

:
I
div_5RealDivmul_16add_11*
T0*
_output_shapes

:
[
sub_10Sublstm_1/recurrent_kernel/readdiv_5*
T0*
_output_shapes

:
Б
Assign_9Assignlstm_1/recurrent_kernelsub_10*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes

:

group_deps_1NoOp^add_1^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
Ѕ
initNoOp^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^lstm_1/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign
^lr/Assign^rho/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign""
	variablesњї
=
lstm_1/kernel:0lstm_1/kernel/Assignlstm_1/kernel/read:0
[
lstm_1/recurrent_kernel:0lstm_1/recurrent_kernel/Assignlstm_1/recurrent_kernel/read:0
7
lstm_1/bias:0lstm_1/bias/Assignlstm_1/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0

lr:0	lr/Assign	lr/read:0

rho:0
rho/Assign
rho/read:0
%
decay:0decay/Assigndecay/read:0
4
iterations:0iterations/Assigniterations/read:0
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0"Ы
while_contextИД
А
lstm_1/while/lstm_1/while/  *lstm_1/while/LoopCond:02lstm_1/while/Merge:0:lstm_1/while/Identity:0Blstm_1/while/Exit:0Blstm_1/while/Exit_1:0Blstm_1/while/Exit_2:0Blstm_1/while/Exit_3:0Bgradients/f_count_2:0Jб
gradients/Add/y:0
gradients/Add:0
gradients/Merge:0
gradients/Merge:1
gradients/NextIteration:0
gradients/Switch:0
gradients/Switch:1
gradients/f_count:0
gradients/f_count_1:0
gradients/f_count_2:0
8gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnter:0
9gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPush:0
5gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc:0
8gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnter:0
9gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPush:0
5gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc:0
8gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnter:0
9gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPush:0
5gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc:0
6gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnter:0
7gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPush:0
3gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc:0
\gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnter:0
]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPush:0
Ygradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0
Bgradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc:0
)gradients/lstm_1/while/add_1_grad/Shape:0
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1:0
Cgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush:0
Egradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush_1:0
?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc:0
Agradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0
)gradients/lstm_1/while/add_2_grad/Shape:0
+gradients/lstm_1/while/add_2_grad/Shape_1:0
Bgradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc:0
)gradients/lstm_1/while/add_3_grad/Shape:0
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1:0
Cgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush:0
Egradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush_1:0
?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc:0
Agradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1:0
)gradients/lstm_1/while/add_4_grad/Shape:0
+gradients/lstm_1/while/add_4_grad/Shape_1:0
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1:0
Cgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush:0
Egradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush_1:0
?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc:0
Agradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1:0
)gradients/lstm_1/while/add_5_grad/Shape:0
+gradients/lstm_1/while/add_5_grad/Shape_1:0
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1:0
Cgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush:0
Egradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush_1:0
?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc:0
Agradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1:0
)gradients/lstm_1/while/add_6_grad/Shape:0
+gradients/lstm_1/while/add_6_grad/Shape_1:0
Bgradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc:0
)gradients/lstm_1/while/add_7_grad/Shape:0
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter:0
Bgradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1:0
Agradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush:0
Cgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush_1:0
=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc:0
?gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1:0
'gradients/lstm_1/while/add_grad/Shape:0
)gradients/lstm_1/while/add_grad/Shape_1:0
Rgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnter:0
Sgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPush:0
Ogradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc:0
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter:0
Hgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1:0
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush:0
Igradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush_1:0
Cgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc:0
Egradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1:0
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape:0
Tgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnter:0
Ugradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPush:0
Qgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc:0
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter:0
Jgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1:0
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush:0
Kgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush_1:0
Egradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc:0
Ggradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1:0
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape:0
Lgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnter:0
Mgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPush:0
Igradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc:0
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter:0
Egradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1:0
Dgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush:0
Fgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush_1:0
@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc:0
Bgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1:0
3gradients/lstm_1/while/clip_by_value_1_grad/Shape:0
Tgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnter:0
Ugradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPush:0
Qgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc:0
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter:0
Jgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1:0
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush:0
Kgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush_1:0
Egradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc:0
Ggradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1:0
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape:0
Lgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnter:0
Mgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPush:0
Igradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc:0
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter:0
Egradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1:0
Dgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush:0
Fgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush_1:0
@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc:0
Bgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1:0
3gradients/lstm_1/while/clip_by_value_2_grad/Shape:0
Jgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnter:0
Kgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPush:0
Ggradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc:0
Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter:0
Cgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1:0
Bgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush:0
Dgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush_1:0
>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc:0
@gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1:0
1gradients/lstm_1/while/clip_by_value_grad/Shape:0
Bgradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc:0
+gradients/lstm_1/while/mul_1_grad/Shape_1:0
0gradients/lstm_1/while/mul_1_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_1_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_1_grad/mul/f_acc:0
2gradients/lstm_1/while/mul_1_grad/mul_1/RefEnter:0
3gradients/lstm_1/while/mul_1_grad/mul_1/StackPush:0
/gradients/lstm_1/while/mul_1_grad/mul_1/f_acc:0
Bgradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc:0
)gradients/lstm_1/while/mul_2_grad/Shape:0
0gradients/lstm_1/while/mul_2_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_2_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_2_grad/mul/f_acc:0
Bgradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc:0
+gradients/lstm_1/while/mul_3_grad/Shape_1:0
0gradients/lstm_1/while/mul_3_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_3_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_3_grad/mul/f_acc:0
2gradients/lstm_1/while/mul_3_grad/mul_1/RefEnter:0
3gradients/lstm_1/while/mul_3_grad/mul_1/StackPush:0
/gradients/lstm_1/while/mul_3_grad/mul_1/f_acc:0
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1:0
Cgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush:0
Egradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush_1:0
?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc:0
Agradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1:0
)gradients/lstm_1/while/mul_4_grad/Shape:0
+gradients/lstm_1/while/mul_4_grad/Shape_1:0
0gradients/lstm_1/while/mul_4_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_4_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_4_grad/mul/f_acc:0
2gradients/lstm_1/while/mul_4_grad/mul_1/RefEnter:0
3gradients/lstm_1/while/mul_4_grad/mul_1/StackPush:0
/gradients/lstm_1/while/mul_4_grad/mul_1/f_acc:0
Bgradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc:0
)gradients/lstm_1/while/mul_5_grad/Shape:0
0gradients/lstm_1/while/mul_5_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_5_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_5_grad/mul/f_acc:0
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1:0
Cgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush:0
Egradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush_1:0
?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc:0
Agradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1:0
)gradients/lstm_1/while/mul_6_grad/Shape:0
+gradients/lstm_1/while/mul_6_grad/Shape_1:0
0gradients/lstm_1/while/mul_6_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_6_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_6_grad/mul/f_acc:0
2gradients/lstm_1/while/mul_6_grad/mul_1/RefEnter:0
3gradients/lstm_1/while/mul_6_grad/mul_1/StackPush:0
/gradients/lstm_1/while/mul_6_grad/mul_1/f_acc:0
Bgradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc:0
)gradients/lstm_1/while/mul_7_grad/Shape:0
0gradients/lstm_1/while/mul_7_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_7_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_7_grad/mul/f_acc:0
2gradients/lstm_1/while/mul_7_grad/mul_1/RefEnter:0
3gradients/lstm_1/while/mul_7_grad/mul_1/StackPush:0
/gradients/lstm_1/while/mul_7_grad/mul_1/f_acc:0
Bgradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPush:0
?gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc:0
+gradients/lstm_1/while/mul_8_grad/Shape_1:0
0gradients/lstm_1/while/mul_8_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_8_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_8_grad/mul/f_acc:0
2gradients/lstm_1/while/mul_8_grad/mul_1/RefEnter:0
3gradients/lstm_1/while/mul_8_grad/mul_1/StackPush:0
/gradients/lstm_1/while/mul_8_grad/mul_1/f_acc:0
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1:0
Cgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush:0
Egradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush_1:0
?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc:0
Agradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1:0
)gradients/lstm_1/while/mul_9_grad/Shape:0
+gradients/lstm_1/while/mul_9_grad/Shape_1:0
0gradients/lstm_1/while/mul_9_grad/mul/RefEnter:0
1gradients/lstm_1/while/mul_9_grad/mul/StackPush:0
-gradients/lstm_1/while/mul_9_grad/mul/f_acc:0
2gradients/lstm_1/while/mul_9_grad/mul_1/RefEnter:0
3gradients/lstm_1/while/mul_9_grad/mul_1/StackPush:0
/gradients/lstm_1/while/mul_9_grad/mul_1/f_acc:0
@gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnter:0
Agradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPush:0
=gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc:0
'gradients/lstm_1/while/mul_grad/Shape:0
.gradients/lstm_1/while/mul_grad/mul/RefEnter:0
/gradients/lstm_1/while/mul_grad/mul/StackPush:0
+gradients/lstm_1/while/mul_grad/mul/f_acc:0
3gradients/lstm_1/while/strided_slice_1_grad/Shape:0
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter:0
Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1:0
Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2:0
Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3:0
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush:0
Jgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_1:0
Jgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_2:0
Jgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_3:0
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc:0
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1:0
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2:0
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3:0
3gradients/lstm_1/while/strided_slice_2_grad/Shape:0
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter:0
Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1:0
Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2:0
Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3:0
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush:0
Jgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_1:0
Jgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_2:0
Jgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_3:0
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc:0
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1:0
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2:0
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3:0
3gradients/lstm_1/while/strided_slice_3_grad/Shape:0
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter:0
Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1:0
Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2:0
Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3:0
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush:0
Jgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_1:0
Jgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_2:0
Jgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_3:0
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc:0
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1:0
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2:0
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3:0
1gradients/lstm_1/while/strided_slice_grad/Shape:0
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter:0
Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1:0
Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2:0
Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3:0
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush:0
Hgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_1:0
Hgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_2:0
Hgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_3:0
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc:0
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1:0
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2:0
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3:0
lstm_1/TensorArray:0
Clstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0
lstm_1/TensorArray_1:0
lstm_1/strided_slice_12:0
lstm_1/strided_slice_4:0
lstm_1/strided_slice_5:0
lstm_1/strided_slice_6:0
lstm_1/strided_slice_7:0
lstm_1/while/Const:0
lstm_1/while/Const_1:0
lstm_1/while/Const_2:0
lstm_1/while/Const_3:0
lstm_1/while/Const_4:0
lstm_1/while/Const_5:0
lstm_1/while/Enter:0
lstm_1/while/Enter_1:0
lstm_1/while/Enter_2:0
lstm_1/while/Enter_3:0
lstm_1/while/Exit:0
lstm_1/while/Exit_1:0
lstm_1/while/Exit_2:0
lstm_1/while/Exit_3:0
lstm_1/while/Identity:0
lstm_1/while/Identity_1:0
lstm_1/while/Identity_2:0
lstm_1/while/Identity_3:0
lstm_1/while/Less/Enter:0
lstm_1/while/Less:0
lstm_1/while/LoopCond:0
lstm_1/while/MatMul/Enter:0
lstm_1/while/MatMul:0
lstm_1/while/MatMul_1/Enter:0
lstm_1/while/MatMul_1:0
lstm_1/while/MatMul_2/Enter:0
lstm_1/while/MatMul_2:0
lstm_1/while/MatMul_3/Enter:0
lstm_1/while/MatMul_3:0
lstm_1/while/Merge:0
lstm_1/while/Merge:1
lstm_1/while/Merge_1:0
lstm_1/while/Merge_1:1
lstm_1/while/Merge_2:0
lstm_1/while/Merge_2:1
lstm_1/while/Merge_3:0
lstm_1/while/Merge_3:1
lstm_1/while/NextIteration:0
lstm_1/while/NextIteration_1:0
lstm_1/while/NextIteration_2:0
lstm_1/while/NextIteration_3:0
lstm_1/while/Switch:0
lstm_1/while/Switch:1
lstm_1/while/Switch_1:0
lstm_1/while/Switch_1:1
lstm_1/while/Switch_2:0
lstm_1/while/Switch_2:1
lstm_1/while/Switch_3:0
lstm_1/while/Switch_3:1
lstm_1/while/Tanh:0
lstm_1/while/Tanh_1:0
&lstm_1/while/TensorArrayReadV3/Enter:0
(lstm_1/while/TensorArrayReadV3/Enter_1:0
 lstm_1/while/TensorArrayReadV3:0
8lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0
2lstm_1/while/TensorArrayWrite/TensorArrayWriteV3:0
lstm_1/while/add:0
lstm_1/while/add_1/y:0
lstm_1/while/add_1:0
lstm_1/while/add_2:0
lstm_1/while/add_3/y:0
lstm_1/while/add_3:0
lstm_1/while/add_4:0
lstm_1/while/add_5:0
lstm_1/while/add_6:0
lstm_1/while/add_7/y:0
lstm_1/while/add_7:0
lstm_1/while/add_8/y:0
lstm_1/while/add_8:0
$lstm_1/while/clip_by_value/Minimum:0
lstm_1/while/clip_by_value:0
&lstm_1/while/clip_by_value_1/Minimum:0
lstm_1/while/clip_by_value_1:0
&lstm_1/while/clip_by_value_2/Minimum:0
lstm_1/while/clip_by_value_2:0
lstm_1/while/mul/y:0
lstm_1/while/mul:0
lstm_1/while/mul_1/x:0
lstm_1/while/mul_1:0
lstm_1/while/mul_2/y:0
lstm_1/while/mul_2:0
lstm_1/while/mul_3/x:0
lstm_1/while/mul_3:0
lstm_1/while/mul_4:0
lstm_1/while/mul_5/y:0
lstm_1/while/mul_5:0
lstm_1/while/mul_6:0
lstm_1/while/mul_7/y:0
lstm_1/while/mul_7:0
lstm_1/while/mul_8/x:0
lstm_1/while/mul_8:0
lstm_1/while/mul_9:0
"lstm_1/while/strided_slice/stack:0
$lstm_1/while/strided_slice/stack_1:0
$lstm_1/while/strided_slice/stack_2:0
lstm_1/while/strided_slice:0
$lstm_1/while/strided_slice_1/stack:0
&lstm_1/while/strided_slice_1/stack_1:0
&lstm_1/while/strided_slice_1/stack_2:0
lstm_1/while/strided_slice_1:0
$lstm_1/while/strided_slice_2/stack:0
&lstm_1/while/strided_slice_2/stack_1:0
&lstm_1/while/strided_slice_2/stack_2:0
lstm_1/while/strided_slice_2:0
$lstm_1/while/strided_slice_3/stack:0
&lstm_1/while/strided_slice_3/stack_1:0
&lstm_1/while/strided_slice_3/stack_2:0
lstm_1/while/strided_slice_3:0
Ggradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1:0Jgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1:0a
-gradients/lstm_1/while/mul_7_grad/mul/f_acc:00gradients/lstm_1/while/mul_7_grad/mul/RefEnter:0e
/gradients/lstm_1/while/mul_3_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_3_grad/mul_1/RefEnter:0
Egradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc:0Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter:0
Egradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1:0Hgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1:0
Ggradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1:0Jgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1:0
?gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnter:0
?gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1:0Bgradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1:0
?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter:0
?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter:0
?gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnter:0
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3:0Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3:0
?gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnter:0q
5gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc:08gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnter:0
Agradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1:0a
-gradients/lstm_1/while/mul_4_grad/mul/f_acc:00gradients/lstm_1/while/mul_4_grad/mul/RefEnter:09
lstm_1/strided_slice_5:0lstm_1/while/MatMul_1/Enter:0
@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc:0Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter:0q
5gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc:08gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnter:0
=gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc:0@gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnter:0
@gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1:0Cgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1:0
?gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnter:0
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc:0Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter:0
Agradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1:0
?gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnter:09
lstm_1/strided_slice_6:0lstm_1/while/MatMul_2/Enter:0
Egradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc:0Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter:0
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2:0Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2:0
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2:0Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2:0e
/gradients/lstm_1/while/mul_4_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_4_grad/mul_1/RefEnter:0
Bgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1:0Egradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1:0
Igradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc:0Lgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnter:0
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2:0Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2:0
Agradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1:0q
5gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc:08gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnter:0
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc:0Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter:0a
-gradients/lstm_1/while/mul_2_grad/mul/f_acc:00gradients/lstm_1/while/mul_2_grad/mul/RefEnter:0
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1:0Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1:0
Igradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc:0Lgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnter:0a
-gradients/lstm_1/while/mul_9_grad/mul/f_acc:00gradients/lstm_1/while/mul_9_grad/mul/RefEnter:0
>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc:0Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter:0Љ
Qgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc:0Tgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnter:0
?gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc:0Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter:07
lstm_1/strided_slice_4:0lstm_1/while/MatMul/Enter:0@
lstm_1/TensorArray_1:0&lstm_1/while/TensorArrayReadV3/Enter:0
?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter:0
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3:0Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3:06
lstm_1/strided_slice_12:0lstm_1/while/Less/Enter:0e
/gradients/lstm_1/while/mul_8_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_8_grad/mul_1/RefEnter:0m
3gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc:06gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnter:0
?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter:0
Cgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc:0Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter:0
?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter:0a
-gradients/lstm_1/while/mul_1_grad/mul/f_acc:00gradients/lstm_1/while/mul_1_grad/mul/RefEnter:0e
/gradients/lstm_1/while/mul_6_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_6_grad/mul_1/RefEnter:0
?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter:0P
lstm_1/TensorArray:08lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0Ѕ
Ogradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc:0Rgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnter:0e
/gradients/lstm_1/while/mul_9_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_9_grad/mul_1/RefEnter:0a
-gradients/lstm_1/while/mul_6_grad/mul/f_acc:00gradients/lstm_1/while/mul_6_grad/mul/RefEnter:0e
/gradients/lstm_1/while/mul_7_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_7_grad/mul_1/RefEnter:0
Bgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1:0Egradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1:0a
-gradients/lstm_1/while/mul_3_grad/mul/f_acc:00gradients/lstm_1/while/mul_3_grad/mul/RefEnter:0
=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc:0@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter:0]
+gradients/lstm_1/while/mul_grad/mul/f_acc:0.gradients/lstm_1/while/mul_grad/mul/RefEnter:0a
-gradients/lstm_1/while/mul_8_grad/mul/f_acc:00gradients/lstm_1/while/mul_8_grad/mul/RefEnter:0Љ
Qgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc:0Tgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc:0Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter:0
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3:0Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3:0o
Clstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(lstm_1/while/TensorArrayReadV3/Enter_1:0
Agradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1:09
lstm_1/strided_slice_7:0lstm_1/while/MatMul_3/Enter:0
@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc:0Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter:0
?gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnter:0
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2:0Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2:0
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1:0Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1:0
?gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnter:0
Agradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1:0
Agradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1:0
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1:0Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1:0
?gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3:0Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3:0a
-gradients/lstm_1/while/mul_5_grad/mul/f_acc:00gradients/lstm_1/while/mul_5_grad/mul/RefEnter:0e
/gradients/lstm_1/while/mul_1_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_1_grad/mul_1/RefEnter:0
Ggradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc:0Jgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnter:0
Agradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1:0
?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter:0
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1:0Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1:0Й
Ygradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0\gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnter:0"
trainable_variablesњї
=
lstm_1/kernel:0lstm_1/kernel/Assignlstm_1/kernel/read:0
[
lstm_1/recurrent_kernel:0lstm_1/recurrent_kernel/Assignlstm_1/recurrent_kernel/read:0
7
lstm_1/bias:0lstm_1/bias/Assignlstm_1/bias/read:0
@
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:0
:
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:0

lr:0	lr/Assign	lr/read:0

rho:0
rho/Assign
rho/read:0
%
decay:0decay/Assigndecay/read:0
4
iterations:0iterations/Assigniterations/read:0
.

Variable:0Variable/AssignVariable/read:0
4
Variable_1:0Variable_1/AssignVariable_1/read:0
4
Variable_2:0Variable_2/AssignVariable_2/read:0
4
Variable_3:0Variable_3/AssignVariable_3/read:0
4
Variable_4:0Variable_4/AssignVariable_4/read:0тU       ЃK"	гЃі}GжA*

lossі>јях       OaО(	Іі}GжA*
	
lr
з#<9АA       ШС	ЪЇі}GжA*

val_lossdе3=З~@       и-	г2Э~GжA*

loss!q=сcT       ЃK"		5Э~GжA*
	
lr
з#<K       йм2	Б6Э~GжA*

val_loss|Юц<ыR       и-	ЃGжA*

lossщ0=SВg       ЃK"	SЃGжA*
	
lr
з#<рwК       йм2	ЃGжA*

val_lossєЌ =е|пO       и-	охzGжA*

lossИд=оп        ЃK"	ћчzGжA*
	
lr
з#<еL       йм2	МщzGжA*

val_lossIё<здћ       и-	ЊPGжA*

loss~=?
%       ЃK"	*ЌPGжA*
	
lr
з#<FfD       йм2	ЎPGжA*

val_loss_X№<DyЛ-       и-	цы&GжA*

lossX=ЕнгР       ЃK"	ю&GжA*
	
lr
з#<ЩьеP       йм2	Уя&GжA*

val_lossб\ё<ПgjЗ       и-	§GжA*

lossm=ю@       ЃK"	>§GжA*
	
lr
з#<70       йм2	y§GжA*

val_losssэР<ІгсЙ       и-	RёвGжA*

lossЬ4 =P       ЃK"	ZѓвGжA*
	
lr
з#<Ъ%Yo       йм2	ѕвGжA*

val_lossуxР<;r.ё       и-	И@ЊGжA*

loss{ј<r{U       ЃK"	еBЊGжA*
	
lr
з#<2.В       йм2	ПDЊGжA*

val_loss_еО<eAn       и-	юGжA	*

lossЁє<Ё*p       ЃK"	GжA	*
	
lr
з#<Tй       йм2	5GжA	*

val_lossФЫИ<{)       и-	ЖіUGжA
*

lossІѕю<Sћy       ЃK"	гјUGжA
*
	
lr
з#<Ь-       йм2	+ћUGжA
*

val_lossТ<м+"c       и-	O-GжA*

loss\яч<­]Ьі       ЃK"	l-GжA*
	
lr
з#<И       йм2	$
-GжA*

val_loss3В<VпO       и-	гGжA*

lossЫх<ЧУтВ       ЃK"	№GжA*
	
lr
з#<Шд       йм2	уGжA*

val_lossuьЇ<ЙЙ7Щ       и-	bМйGжA*

lossрц<wЙE       ЃK"	sОйGжA*
	
lr
з#<о En       йм2	#РйGжA*

val_lossЇ<zp)       и-	ПБGжA*

loss0oр<hlь       ЃK"	дБGжA*
	
lr
з#<bр"Д       йм2	ВБGжA*

val_lossяЈ<ѓxD       и-	aрGжA*

loss6Wр<qZv       ЃK"	тGжA*
	
lr
з#<дi)       йм2	~фGжA*

val_lossіР<=Е       и-	^GжA*

lossHЦо<фгдЙ       ЃK"	)^GжA*
	
lr
з#<<
       йм2	^GжA*

val_lossП<б2Q       и-	6GжA*

lossЯл<ЛУЮ       ЃK"	Щ 6GжA*
	
lr
з#<~}F       йм2	#6GжA*

val_lossWќЁ<ъЊЩЬ       и-	2ЦGжA*

lossщл<fг       ЃK"	TШGжA*
	
lr
з#<ЄКRM       йм2	ЪGжA*

val_loss8@Б<Чl       и-	ЁLсGжA*

lossи<&жњF       ЃK"	ТNсGжA*
	
lr
з#<
ўС       йм2	PсGжA*

val_lossЁ<чсћВ       и-	KБЗGжA*

lossїй<дялП       ЃK"	}ГЗGжA*
	
lr
з#<K"І       йм2	ШЕЗGжA*

val_lossj,­<Юk5       и-	-GжA*

loss]к<~Zы-       ЃK"	Ѕ/GжA*
	
lr
з#<В:p       йм2	b1GжA*

val_lossEйБ<цЊ7       и-	љdGжA*

lossж<6Ж"       ЃK"	dGжA*
	
lr
з#<n)г       йм2	]dGжA*

val_loss&Л<Зќg       и-	k+:GжA*

lossз<вД­       ЃK"	-:GжA*
	
lr
з#<Ђ`a       йм2	j/:GжA*

val_lossл<яM
o       и-	BьGжA*

loss§~а<FЏOљ       ЃK"	tюGжA*
	
lr
з#<+Й       йм2	№GжA*

val_loss}XЁ<љљ       и-	.цGжA*

lossжН<<^OЪ       ЃK"	\цGжA*
	
lr
з#<Y<Щ       йм2	KцGжA*

val_lossRѕ<=Ќгў       и-	%МGжA*

losspЕ<ЈSч       ЃK"	'МGжA*
	
lr
з#<ъ-я       йм2	д(МGжA*

val_lossАЋ<UЎz       и-	ЛGжA*

lossТPЗ<ГбvL       ЃK"	ЯGжA*
	
lr
з#<ЦН       йм2	О GжA*

val_loss*B<       и-	&ЧgGжA*

lossЬkЌ<bџђc       ЃK"	?ЩgGжA*
	
lr
з#<г*yy       йм2	ЫgGжA*

val_lossЭu<чSkТ       и-	??GжA*

loss>Ѕ<Ff       ЃK"	еA?GжA*
	
lr
з#<ќЮбж       йм2	ФC?GжA*

val_lossЖ<ьР       и-	ГxGжA*

lossmX<uЛ@w       ЃK"	йzGжA*
	
lr
з#<Щ[Й       йм2	}GжA*

val_lossз<Ј!]        и-	љKъGжA*

lossЇF<Л\ЭО       ЃK"	NъGжA*
	
lr
з#<$шгk       йм2	пOъGжA*

val_lossр*0<Чo       и-	СGжA *

lossFє<$ЎД       ЃK"	СGжA *
	
lr
з#<РjЃ       йм2	цСGжA *

val_lossRJ.<№|Р       и-	хGжA!*

lossj<нux!       ЃK"	љGжA!*
	
lr
з#<wїЙ"       йм2	КGжA!*

val_lossюр+<qД/	       и-	hkGжA"*

loss*I<уЂ       ЃK"	tkGжA"*
	
lr
з#<Сш       йм2	[
kGжA"*

val_loss9]<эЬЩ       и-	2щ?GжA#*

lossBу?<KКї       ЃK"	Fы?GжA#*
	
lr
з#<hњъ       йм2	э?GжA#*

val_lossUЁ#<`зj       и-	ИGжA$*

lossЎ*;<JЙm       ЃK"	б!GжA$*
	
lr
з#<Д       йм2	Г#GжA$*

val_loss>ђ;Ёkз       и-	GщGжA%*

lossE<ЅБ       ЃK"	JщGжA%*
	
lr
з#<GaM       йм2	QMщGжA%*

val_lossEёe<eВе       и-	GчПGжA&*

lossK§<ф       ЃK"	6ъПGжA&*
	
lr
з#<W?Ш       йм2	NьПGжA&*

val_lossМъ<KU<ї       и-	-
GжA'*

lossЎ{<L№3$       ЃK"	RGжA'*
	
lr
з#<ьstь       йм2	дGжA'*

val_loss%и<(єп       и-	рesGжA(*

lossё<hЩЎh       ЃK"	hsGжA(*
	
lr
з#<h"@       йм2	YjsGжA(*

val_lossЦєY<^vЂ       и-	ЙH GжA)*

lossjЩ<ЅKЧ       ЃK"	ВЛH GжA)*
	
lr
з#<wl|Њ       йм2	sНH GжA)*

val_lossШ <m       и-	Я0ЁGжA**

loss <Ђ4Й       ЃK"	з2ЁGжA**
	
lr
з#<M4       йм2	з4ЁGжA**

val_lossћгј;zРh       и-	§ёЁGжA+*

loss$M<ЃИ4п       ЃK"	ХџёЁGжA+*
	
lr
з#<зA       йм2	ођЁGжA+*

val_lossAО;h0       и-	ШЂGжA,*

loss(<1P>       ЃK"	ДШЂGжA,*
	
lr
з#<љЅ       йм2	hШЂGжA,*

val_lossуЦЇ<еК       и-	:ЃGжA-*

lossн<тЌкE       ЃK"	щ
ЃGжA-*
	
lr
з#<ГUЙѕ       йм2	,ЃGжA-*

val_loss№Л< о       и-	qЄGжA.*

lossd<Zю       ЃK"	qЄGжA.*
	
lr
з#<Цo       йм2	SqЄGжA.*

val_lossњ<С       и-	УEЅGжA/*

lossЅШ<Rја       ЃK"	7ХEЅGжA/*
	
lr
з#<Z!у"       йм2	eЧEЅGжA/*

val_lossY`і;лО       и-	^ІGжA0*

lossYў<amЕ       ЃK"	ІGжA0*
	
lr
з#<№       йм2	ТІGжA0*

val_lossjxЌ<2u       и-	їяІGжA1*

loss§В<ц       ЃK"	љяІGжA1*
	
lr
з#<1ЇF       йм2	zћяІGжA1*

val_loss1"Y<О       и-	шиФЇGжA2*

lossѕ><ЃН-
       ЃK"	№кФЇGжA2*
	
lr
з#<ѕuE       йм2	+нФЇGжA2*

val_lossE2<@v+?       и-	QgЈGжA3*

lossЄK№;оЌO       ЃK"	iЈGжA3*
	
lr
з#<С       йм2	ЦkЈGжA3*

val_loss>'<Ѕ>       и-	UoЉGжA4*

lossЄ[<Юцє       ЃK"	ЃWoЉGжA4*
	
lr
з#<|ц|R       йм2	SYoЉGжA4*

val_lossЕЫ<TѓД[       и-	ХЦDЊGжA5*

lossЉв<5SЦз       ЃK"	*ЩDЊGжA5*
	
lr
з#<-ш       йм2	KЫDЊGжA5*

val_lossп­;l       и-	рЋGжA6*

lossg§;ж^юu       ЃK"	3тЋGжA6*
	
lr
з#<gЪ       йм2	фЋGжA6*

val_lossІ`7<тА7       и-	,+юЋGжA7*

lossаc <3]Нє       ЃK"	A-юЋGжA7*
	
lr
з#<zmоy       йм2	#/юЋGжA7*

val_lossш- <+фл       и-	.ХУЌGжA8*

lossZь;u%щ*       ЃK"	PЧУЌGжA8*
	
lr
з#<dVДШ       йм2	ЩУЌGжA8*

val_loss^ Љ;ь~ї       и-	t­GжA9*

loss+я<K1уи       ЃK"	.v­GжA9*
	
lr
з#<D       йм2	`x­GжA9*

val_loss<qqА#       и-	9oЎGжA:*

loss7Rџ;тІ9       ЃK"	IoЎGжA:*
	
lr
з#<uн       йм2	ЁoЎGжA:*

val_lossLG<5gВП       и-	ЙсDЏGжA;*

lossуђ;.~>       ЃK"	ЮуDЏGжA;*
	
lr
з#<#	юЙ       йм2	хDЏGжA;*

val_lossmEЄ;Ї*lЩ