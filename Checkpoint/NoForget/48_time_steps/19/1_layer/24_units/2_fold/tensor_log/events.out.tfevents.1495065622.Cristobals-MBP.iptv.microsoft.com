       �K"	  �9G�Abrain.Event:2t��qq	     �Em	�۠9G�A"�
b
lstm_1_inputPlaceholder*
dtype0*
shape: *+
_output_shapes
:���������0
l
lstm_1/random_uniform/shapeConst*
dtype0*
valueB"   `   *
_output_shapes
:
^
lstm_1/random_uniform/minConst*
dtype0*
valueB
 *O�~�*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
dtype0*
valueB
 *O�~>*
_output_shapes
: 
�
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
dtype0*
seed2���*
seed���)*
T0*
_output_shapes

:`
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
T0*
_output_shapes
: 
�
lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
T0*
_output_shapes

:`
{
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes

:`
�
lstm_1/kernel
VariableV2*
dtype0*
shape
:`*
	container *
shared_name *
_output_shapes

:`
�
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes

:`
x
lstm_1/kernel/readIdentitylstm_1/kernel* 
_class
loc:@lstm_1/kernel*
T0*
_output_shapes

:`
T
lstm_1/SquareSquarelstm_1/kernel/read*
T0*
_output_shapes

:`
Q
lstm_1/mul/xConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
W

lstm_1/mulMullstm_1/mul/xlstm_1/Square*
T0*
_output_shapes

:`
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
�H
%lstm_1/recurrent_kernel/initial_valueConst*
dtype0*�H
value�HB�H`"�H��F��]�=���9��=,�U����C�>(��}�P���&>5G,��R�=蓳=M�P���V�]脽
g�;Ѝ	=�xb������%�0½�����0�[1���Ƚɋ����C>. �=�B<S�=���*=�8j�O����μ�u�K�=��f<RH�,�n�x�.>B�6=�K�<��h�=�P>B'���>���E>c�!�������>��=:����=!LO�?��ƧڼO	�==�=��#�!uP�Z ��J�4>3q@=	��<�]ʼe��=+�<<�����S����/:0��y�� �?��[+�	n��RP������y�?w">%�V�=�1��g9���F=����V�:n�ͼUO��:U�:
*���G>�1�$�޼��Ľe�v���=qq�=µ�����<9��:m��|#�+�?W�z�l��н*꼅~H>z�$�Fጾ�8M<a���gO,>���=ʽ}�=@؎��c�=�^>��v{=�YR=�B�;��<��=��=Lba�d�>�=�0��� ��-ռ�o�=Y|����<�R>iٽ�r>єC��刽��Z�m�p=�Ƚ�=�3�<K&�G�r��	U�B��=z{\�;¼��;P��"�f=
>���|��A�%>=��N�=�`�=�㱽$6=<w>�c��z�h��=H�Z���S��.I��r�=`J��J�̽=��=�N�`��;K�*�46����;㬬�*T���p�J.��>��=����A/+��E�=xa=��=I3�=w+�=�.|�6x>qjE=G�<eE@�>o�== ����J�s>�>	�㽟W=�1�=����(y�0�o=~n�<���<*�t�2Є=�p:�,�<ԥp=�=��>�6=���v�z����н/"��+6!�ʁ.������|��6����;������p�ս�=?>i��=�g�e�=��=�,�ː����=? =}K�=�a���΅<�� ��LӼ9��=q�=�}<�&��&>��3:`*S��Ǚ��O=ƿ3�ň�=Q:�>�+�=� ֽBT{��!G�`����=&��f��F~=ݼ���󥽡�(��ڻ�V����A #>��۽Se�c1=�M<�G����<��&���=�Zu�/�=wü�F���K[ۼh&�=,ޡ=]E��ɳ�IC\���>��%>Uzj<��H���<_�лܻ�FW#=9*�=d:��i7�����������=J� =�r�=&/K��N<=o���ד<��>�n=?�Y�v��=��$>����`�=U�ƽ�î��>j�༢~�=m>�Y=v�=�q�=���:��=�<��_����=�8��K���.N<^WǼ�`�<l�=M�=����!=㶡=F��˓ >�d��'9�񘉾xؼ-`!>xI����U=�5�Ŋ>NĐ=e����Hq��-�3D= Ԡ=��e=�[ǽ����n�=�$>w�=0���ɲ!�]��8t�<��=��=!7�B8��S�����=3�d�_�f<]��"[2>gM#�^h�;����@s�s�?�0��=��>	��<�j�=|����>�9=���?�B��v<҄Ƽ+a=�%�}M�<�@3���=�b��A=��
�rz=)n)=Y��C>c�r���K>���d�ȕ��ۓ=3�<�j9�~�=�~�=����zhj�>G =e���g�=,��>m�=/�=Y�=��_��N=��;ϐ�=A�><[�=/�h���)="a��t�M>���= ��<�ծ=��l���<Y.�u;��]>MF;�$(�=k����=�G"�
׌>��\=+T�=��8����;I���j,=����5�x�Q��d�<QO�<�&=�b��m��Di����=��Z<Z�U���Q?�=X���мIB�`j)=N�=���Is���>�w&��U�=z��<Gnm�y�S,�:Y��=2��?H�=}8a�7P�>��f�a���r�9>I����=e}d= ��<<Y"�rY}=6<�ߑ�[��<��;���w,;��½����;��By��cf�<��;&޽��ֽUE�=�K>��=�f%��	>�q�ĝ��Җ>9	�=y�:>`e�;S����=U�H�'�<q��i�=u�����>#��>����R�=f�>���=��=q�?>�7;=��;���<�
�=��<���=��<��=��&�WW}�b��=��Q=�\���Ds���߽j >�����O��'�a=-�g<8b >N��-8��B����fǉ>}:L��Q��@	>��=�	>��.>�)>���6B���r=�����B�<b��=��]�._ƽ=�a�1G;wb�L��Ͻ��<��->8H�=W���L��ٚ=�ћ��(>��9��ǵ;^��"3��M�ݽ�����R#�$X>�;�=�C����W�B�(ڜ�m��=�SS���==t5��F>ͼF��;���<����������j��Y����=�k\>���~�˾�[���e<Uj<�T��s����$���ܽ�/����=��v=�4�=t$>�����1�l�н���<�-�<p�=�b=��V<�?���KF>��L��5>�4X����sӦ=8ڣ=L%�=�緽1�=Щ�
�#=<��;o�=Cv6>���x��d�<h>�O�?H={L���=5A�:�b>�	>��>��>����Pz�= E���=��Q1�+�4<�J���>��D=c�E���"�6_���p�=X�=7Y<�OG��>���[g��;�A�=�0潟����h=u�k=ψ,�]/>i>�㌼�9n�z�<q ��ܨ;{"�gF������=����>�_[=͍�RR�����kWʼ�aܽ��>�=�n�=�p:����I����
P=z�k�l+���y����=vk��������=�[�=;�ӽj���i�=���b:��W�+�u&1��ؽ��;��>���=�-9���<�佝��<�>-Wo=բI=��l=��r�7 �=�`������h=����Uy=�V�K]=ֳ��s'�c�>wK=�V�.3>x�]�!��K�=n�X�$�8<r���"ܽ��Y=��o�B�?>հV�H�3=�`�=K'�=˛q=��M��s>Cm�=��G�Kb]=��&>m`d��g����B�PP���`M>�e��8=P�� ˼���;��<�-�;U���g=�^�<���;!<�*�<���=c}��W��<,��<L�3�t=�Kt=2����<T&��lƨ=���<�d��P���-���>%�Gd��j��H���ǅ��6=Ӵ���K��*��=����T��=�[-�Fm=�d�=U`D=M�(�����f�~�ؽ��=�@
�#+�A��1>�f>7z���:���:01���_���>���;��̼h�ܽwB_�/"�����(G���c��ޞ�;��:=�n���=!�=o߼�2 *=N�=
�V���ݽ�#�=ӳs�� �=&?ؼ�=Y�ܽ��6'F=Q�����Z�<`�<=<DA�6ݯ=B>=�Q#>t�Y>�[��f��=��>G>Ƚ��ӽ.Ky�]ia������d�����+�<�k���<�k�->!�������A�Ƌ{�q����fJ=��m��j���c*���������8M>�>�1�uн~�=vet��o�=��=�;=��<���<Yg�=C��<i�=�r���9`>��U<H�/<��=��:��!o=�>#���r��B`&<	*��f�ѳ�=c=2�>�Մ<jN۽ �8��>~�=?�O��%���զ�|,�=y���\Ȑ�";"x�mQ������R+ڼ~0����>pѽX�W���x�!J�<��D��_=n�-8�l��㮽����>�;=�,?�hJO�S'C��4�l����ʽ
�*>�$$=z�T��U���l=�`���X=���=c˹=<��=*2;���;H�<���=�Jh�A��=dL/��T =���m���� Y��8����j�"X��/��d�|>&�=��;>O�۽Hэ�ݜ�*(����@/f<�糽i�=���<�d�=wG�=A���q��7>��'>���=���9���=UB��r�=G�)>f#�[��<��=����F^��y�=��콎��<��g�=����󽽓�=P�F>��=K��o'<�����`�=�倽r����I��^����=�C�=�ㅻ��k<��=>�����P=1+s>��p�`v�=\_=Z�">��~�=/>;°<T���ۻ�@�x=tz=kB=v�w���=�L5�S�c=?�v=�A�f�@�AHv>:�S<�=��w����<eBn=��e=�vo��B�<�5�<���}�<C�<�5�=�rF=��Ǽl�=/��;��'>�煽a�<=�&�=�1>�w~=�<:=No�;�r���r$=�~��Cܽ�̽eb0>ߦ<4��鱫�mf��0S���Խ����L���<=<�k�Hq =3������<�=�<sj�=�9���b�?�n�i��;F��=��U�b�F�m]=jl��򣘽�,�=��K���'��u=�!4>�{����q=B�U=hr�<?Ah���y�DV=�ʸ=6��΍�=G`s>�>�ω�,�@�=l�����BO>�ѻ�f=w>;|�-=h��.�<�9�=mLt��d����;<&��=U���L����>���<<}�G��
`�=�]���D=���=y�"�`���2�w9���C>s!�=�ə=�;�<g���o=B8 ��� >��,��ӑ��:�=L�����L>���<ȧ�=˾[�ʦ��|�!�G>RH\=��<=�9Z<���<�|P���'p�<K.P�����z���q��=|�D=V��ڪ>8�
�(�"��n���o0>['ѽ>�Y�^j�=Tֵ=�Q���=۽X�X���>%����@�=�E���*ʽ-�;��q�=��ݽř��<�=#'�=mʸ��G���0��[���'/��|�=1����)�;�O�<�g=(­�-�[���>�r�|;������짻�� w<?x�=��`=Ȃ~=��2����"��+���?>Ο�=�����3�=�����x�=Wv�=0U�+�$=�t���Y�=3a�=p�z=x�H=��>�U=����	\=|�w�c�D>�,�=��=V�#�v�%����D�=>�a� ΋�� �=t<��;��C=7k��_f�=�1)>���=u5��Y� PF�CMỻf��������I�v=u��=4[>��<o� �	0=ς���f<�kN�X!7�*�=T ���m���<B6�=2�D���>𮩽�>n��=�0�(�����y���<Z=��m�����p>��=����T>�]R���l=�l��iB�=/B�=*(�=�>�	�=�>��K�Eｧ])=-���}{���ͽޗC�K��ғ�=c`-��j<h����e=Z�4=�)>�Ҝ<�m,��D��,�:\���=-|=LTO��a��W=�ٚ=���=|M>����K���m�9�����2>a����]=񉒽���v[�=R-���/�ي2=��=��=��� >���Hĸ=�����=My��x�M�ҽF}߽��<�U�=6�.=�������=��1>��!�x��=(<~�խ
>`G/>�5t<焀��,�&^[=�3�<c��|�=V�=c������Og >��	=@k���L!��_(���>'mN��W>_3�����)/�_�J!�=�d����h��=��d>���=@h�=��U>3�4=�[=���=K>:*�O����M�=(�f�L������G6=�Ƽ�^>k)޼���=.t���>�ý�;�=�����>��<�~1=�3�c~Q���ǽ�6�=� �ҙ�=6��!]�=+:�=�R=^X��]��wɑ�.��<�,>��6���Z�D|��@�=|Y<��>�י>��1b<�?�<\����vE��d>�b�=�˽V�U=4F�v�J���<�; �+��=sj=j!�9�5�=LC���="4>���=�q����9>���:o���o]^����=�C�w����=�-�Z�>Q$�=�VۼTd���O>V�W	\��&
�ɓ#�%Yv=k����L��/V�������������m�=
՗<���`��<i��<��2=6�=&>����I��x�<��<=�F���X<�S����<�= > 8齶I����èh=`���U>W��L��<-�R��$�=�M=�E��_�l�=]�=:5꽓�>Y���du�}�M�ur>�����E�ӂ� �=kgC�G� >x
�=>��Li3<Ϲ0����=ts>ļH=��=��O<�<2�߼i��!��=9tj>������=�W�=�"��K>��<� �M[S�B
 >x�����9>��:��P�=|]V>��
�\*�=�:.�?og�1~->K��BI�;�B�0�=4ـ���	>wZ8��㼞��=m�>��F>2����8��V~:; ����x�b�7�;2mr=?M�<�}:��һy�>�r�<U�p=aMV<��=���ߤ��}������0�<�"���qG=:�A>-��<� �cbd�*�>W����j�Z�6�P��r�]=��=t����0�ۣ?�Y*��o�2��V'��#*>���� �^<U&���= 駼F��=�1#�����3��� <��<����0��]Y�=(�� J�=8O>~Z$�yJ�Nlb� ·�|-��"�=��;����9�=�ڍ=��s��H>���=��s����׾=ϧd�1����=�b>hm����'>���:��	�6�M<�`��oQ>�>�V�=f��;����=H#>!��=h�V�oZ��	�=^�P�j�r�N��K�@�^;���,e�=�&>{C�;iQs�0u�=�+/�Uܲ���2���9��=:Uн^31=�hN� ������$��6�=�=r��.\(=?;�9��=��ý�Ҽ������۽�3O<5�S=۟|�H~:�;;��S�<�����!�sN8<���^�=KZ>��	��8��g.=��Nl��*<=��=� �=�Aa<���b[:=d�����=Uj�=�.�<�+=yڔ=S�bU��re9&!ҽ l��(2=n:H=.5�<(D@�H�=�%<hBǽ'D�;�=�� ��=(��aػ@(�>Vx>�޽k�=�T>w{E=|���J#��ؗ=/1�=���<Dռ D>��&�<��)�:>�^�=3��=�f���G��h��49E>7h�;]�;�<��,���4�	�
=u0
��P�=��<�\T��F>�}�=���=��'=�#�=�/�=)%�>W�}<�	>��޽�"k<n#>�	���=ɨ�;���=�O=ڈs�
�v=��=)�&=)Z#�[}j�H�ս�>���c��=�fi=�[��3�'��	�=��=<Y��5�p=�Z�#�;�J�2=��ĽL��0��=����4��=��H>	u.>Ag=uY�=����,�^=%�r��a=%����ݽ4$>�<�=�G=(=#R2���>���@�>�q����<xN�;(��w�=��� ̇=a��<MWa�ך�=�w��_%=>~��J��vX�=��Ž�=�cͼ��<��=X">�=U�N>�x���U�;Pý��=l��aQy�ma�<���`^k>���=�J!>S���ςc�"�y.��rÉ���r>گ�<��2����D��=$��@6���6=���=g��;`r����	<�,��Z�ս긽4�ý#��=��G��!>�a>�$>�� �<�:�[JѼ�(�=�[=q�ļ��T=^��=T�1�J�2>j�Y�+i��ռ'��<m�o��H�<��=���=Z�"/K��΁=�%�=e�����=�KT�z��=��<��
<��i�BKd��$^=��<�?�<�C�$V�]ҽ#�ͽAO�=P�=��<מ�=��=�!�9�=�����+>7A�<���<�(l�xSI�2q��}q>� Լ�`m=:𫽄�=�=zbK<U�sV�=�擽}���#���-�$M=z�0=-1�����$���Z"<�6>�����j��IL<"+(>�ý�0';+\�=6�,��V�P��=6�ٻ42]=�~�=mE���d%=����>����B=h�ͽu���m=�ko�;;r#>`G=�k>�ռ�܌��1�<D=�Gy=\�=�[̼��<�|��.�O>z	�;㹹<�p=�z8�Uh�=h�$=�gн(�޺��:��^.=�*��>]�b��gr��o2<̽M�5,�<�N���<�f��<������������<v��=0Mݽ�F�=���=�|�=J�*������=�U@��޼0�=c1�<ue=�Zh�J�4��<�<��[>����=��=���=&�7>X�=>.l'>0���;��#�ٽ�P=���=|�ƽw,���*��A&>�-G=S�qOH��xd=6)>���<A����6�=��==�A�y��<��E��醽���=��,�(L�q)�:��� +���3���>e�@<��=5�O=0���)=̬b<�]u���B=Q³�Q�P�a�=�0���=�q�=y0�Q��f��T�>ｼ=�H��G�6>�G�2�ﶋ=]<>���=���=�Žoo�=u[��OA��>��Լ6$�=!J�=�D�<��I�K�*>���<���=��%> �>�A;>��ؽ��㽴<��S&�mWA�
4�~^�WV��7��x=�<=s��;�� =��:X=�;���y*=>-���N&�����=@�j���v!
�:'���� �=��:�p8q�j<�=��j����=+���-5��_=������=��E�A�8<�&n�2�3�Ƿx��x�޽�(D�is�]�=���l���������o��V��$Xǽ.[`=�/�=F��譢=E�<U;�=����}�=��4�����I��<���Fs����=2v����?>�$#�g��=���=_ֽs^=�k�S]=&=W�I�V>l��=w����(�iڽh2�<�'>d�=.��=�,���>�&>k��=,݄=��ٻ��Q�ف�< m=%t��r#r��x>�ϻ���$�ǹ;<��=��q����Kܚ=^d�={@��
��Ρ��Ƅ���*
_output_shapes

:`
�
lstm_1/recurrent_kernel
VariableV2*
dtype0*
shape
:`*
	container *
shared_name *
_output_shapes

:`
�
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes

:`
�
lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
T0*
_output_shapes

:`
[
lstm_1/Const_1Const*
dtype0*
valueB`*    *
_output_shapes
:`
w
lstm_1/bias
VariableV2*
dtype0*
shape:`*
	container *
shared_name *
_output_shapes
:`
�
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/Const_1*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes
:`
n
lstm_1/bias/readIdentitylstm_1/bias*
_class
loc:@lstm_1/bias*
T0*
_output_shapes
:`
k
lstm_1/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
m
lstm_1/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
m
lstm_1/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_sliceStridedSlicelstm_1/kernel/readlstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_1/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_1Const*
dtype0*
valueB"    0   *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_1StridedSlicelstm_1/kernel/readlstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_2/stackConst*
dtype0*
valueB"    0   *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
dtype0*
valueB"    H   *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_2StridedSlicelstm_1/kernel/readlstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_3/stackConst*
dtype0*
valueB"    H   *
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
�
lstm_1/strided_slice_3StridedSlicelstm_1/kernel/readlstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

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
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_4/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_4StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_5/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_1Const*
dtype0*
valueB"    0   *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_5StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_6/stackConst*
dtype0*
valueB"    0   *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
dtype0*
valueB"    H   *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_6StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_7/stackConst*
dtype0*
valueB"    H   *
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
�
lstm_1/strided_slice_7StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

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
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
lstm_1/strided_slice_8StridedSlicelstm_1/bias/readlstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
f
lstm_1/strided_slice_9/stackConst*
dtype0*
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
dtype0*
valueB:0*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
lstm_1/strided_slice_9StridedSlicelstm_1/bias/readlstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_10/stackConst*
dtype0*
valueB:0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
dtype0*
valueB:H*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
lstm_1/strided_slice_10StridedSlicelstm_1/bias/readlstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_11/stackConst*
dtype0*
valueB:H*
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
�
lstm_1/strided_slice_11StridedSlicelstm_1/bias/readlstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
b
lstm_1/zeros_like	ZerosLikelstm_1_input*
T0*+
_output_shapes
:���������0
o
lstm_1/Sum_1/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/Sum_1Sumlstm_1/zeros_likelstm_1/Sum_1/reduction_indices*#
_output_shapes
:���������*
T0*
	keep_dims( *

Tidx0
`
lstm_1/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
lstm_1/ExpandDims
ExpandDimslstm_1/Sum_1lstm_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
f
lstm_1/Tile/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:���������
e
lstm_1/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
}
lstm_1/ReshapeReshapelstm_1_inputlstm_1/Reshape/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
lstm_1/MatMulMatMullstm_1/Reshapelstm_1/strided_slice*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*'
_output_shapes
:���������*
T0*
data_formatNHWC
a
lstm_1/stackConst*
dtype0*!
valueB"����0      *
_output_shapes
:
}
lstm_1/Reshape_1Reshapelstm_1/BiasAddlstm_1/stack*+
_output_shapes
:���������0*
T0*
Tshape0
g
lstm_1/Reshape_2/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
lstm_1/Reshape_2Reshapelstm_1_inputlstm_1/Reshape_2/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
lstm_1/MatMul_1MatMullstm_1/Reshape_2lstm_1/strided_slice_1*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*'
_output_shapes
:���������*
T0*
data_formatNHWC
c
lstm_1/stack_1Const*
dtype0*!
valueB"����0      *
_output_shapes
:
�
lstm_1/Reshape_3Reshapelstm_1/BiasAdd_1lstm_1/stack_1*+
_output_shapes
:���������0*
T0*
Tshape0
g
lstm_1/Reshape_4/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
lstm_1/Reshape_4Reshapelstm_1_inputlstm_1/Reshape_4/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
lstm_1/MatMul_2MatMullstm_1/Reshape_4lstm_1/strided_slice_2*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*'
_output_shapes
:���������*
T0*
data_formatNHWC
c
lstm_1/stack_2Const*
dtype0*!
valueB"����0      *
_output_shapes
:
�
lstm_1/Reshape_5Reshapelstm_1/BiasAdd_2lstm_1/stack_2*+
_output_shapes
:���������0*
T0*
Tshape0
g
lstm_1/Reshape_6/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
lstm_1/Reshape_6Reshapelstm_1_inputlstm_1/Reshape_6/shape*'
_output_shapes
:���������*
T0*
Tshape0
�
lstm_1/MatMul_3MatMullstm_1/Reshape_6lstm_1/strided_slice_3*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*'
_output_shapes
:���������*
T0*
data_formatNHWC
c
lstm_1/stack_3Const*
dtype0*!
valueB"����0      *
_output_shapes
:
�
lstm_1/Reshape_7Reshapelstm_1/BiasAdd_3lstm_1/stack_3*+
_output_shapes
:���������0*
T0*
Tshape0
T
lstm_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
lstm_1/concatConcatV2lstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7lstm_1/concat/axis*
N*

Tidx0*+
_output_shapes
:���������0`*
T0
j
lstm_1/transpose/permConst*
dtype0*!
valueB"          *
_output_shapes
:
�
lstm_1/transpose	Transposelstm_1/concatlstm_1/transpose/perm*
Tperm0*
T0*+
_output_shapes
:0���������`
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
�
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
�
lstm_1/strided_slice_13StridedSlicelstm_1/transposelstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

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
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_14/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_14StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_14/stacklstm_1/strided_slice_14/stack_1lstm_1/strided_slice_14/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_15/stackConst*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_1Const*
dtype0*
valueB"    0   *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_15StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_15/stacklstm_1/strided_slice_15/stack_1lstm_1/strided_slice_15/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_16/stackConst*
dtype0*
valueB"    0   *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_1Const*
dtype0*
valueB"    H   *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_16StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_16/stacklstm_1/strided_slice_16/stack_1lstm_1/strided_slice_16/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_17/stackConst*
dtype0*
valueB"    H   *
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
�
lstm_1/strided_slice_17StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_17/stacklstm_1/strided_slice_17/stack_1lstm_1/strided_slice_17/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
S
lstm_1/mul_1/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
b
lstm_1/mul_1Mullstm_1/Tilelstm_1/mul_1/y*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_4MatMullstm_1/mul_1lstm_1/strided_slice_4*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
lstm_1/add_1Addlstm_1/strided_slice_14lstm_1/MatMul_4*
T0*'
_output_shapes
:���������
S
lstm_1/mul_2/xConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
c
lstm_1/mul_2Mullstm_1/mul_2/xlstm_1/add_1*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_2lstm_1/Const_3*
T0*'
_output_shapes
:���������

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_2*
T0*'
_output_shapes
:���������
S
lstm_1/mul_3/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
b
lstm_1/mul_3Mullstm_1/Tilelstm_1/mul_3/y*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_5MatMullstm_1/mul_3lstm_1/strided_slice_5*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
lstm_1/add_3Addlstm_1/strided_slice_15lstm_1/MatMul_5*
T0*'
_output_shapes
:���������
S
lstm_1/mul_4/xConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
c
lstm_1/mul_4Mullstm_1/mul_4/xlstm_1/add_3*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
y
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/add_4lstm_1/Const_5*
T0*'
_output_shapes
:���������
�
lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_4*
T0*'
_output_shapes
:���������
j
lstm_1/mul_5Mullstm_1/clip_by_value_1lstm_1/Tile*
T0*'
_output_shapes
:���������
S
lstm_1/mul_6/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
b
lstm_1/mul_6Mullstm_1/Tilelstm_1/mul_6/y*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_6MatMullstm_1/mul_6lstm_1/strided_slice_6*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
lstm_1/add_5Addlstm_1/strided_slice_16lstm_1/MatMul_6*
T0*'
_output_shapes
:���������
S
lstm_1/TanhTanhlstm_1/add_5*
T0*'
_output_shapes
:���������
h
lstm_1/mul_7Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:���������
a
lstm_1/add_6Addlstm_1/mul_5lstm_1/mul_7*
T0*'
_output_shapes
:���������
S
lstm_1/mul_8/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
b
lstm_1/mul_8Mullstm_1/Tilelstm_1/mul_8/y*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_7MatMullstm_1/mul_8lstm_1/strided_slice_7*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
lstm_1/add_7Addlstm_1/strided_slice_17lstm_1/MatMul_7*
T0*'
_output_shapes
:���������
S
lstm_1/mul_9/xConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
c
lstm_1/mul_9Mullstm_1/mul_9/xlstm_1/add_7*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
y
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/add_8lstm_1/Const_7*
T0*'
_output_shapes
:���������
�
lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_6*
T0*'
_output_shapes
:���������
U
lstm_1/Tanh_1Tanhlstm_1/add_6*
T0*'
_output_shapes
:���������
m
lstm_1/mul_10Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:���������
�
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_12*
_output_shapes

::*
dtype0*
dynamic_size( *
clear_after_read(* 
tensor_array_name	output_ta*
element_shape:
�
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
�
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
�
lstm_1/TensorArrayUnstack/rangeRange%lstm_1/TensorArrayUnstack/range/start'lstm_1/TensorArrayUnstack/strided_slice%lstm_1/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
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
�
lstm_1/while/EnterEnterlstm_1/time*
_output_shapes
: *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
�
lstm_1/while/Enter_1Enterlstm_1/TensorArray:1*
_output_shapes
:*
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
�
lstm_1/while/Enter_2Enterlstm_1/Tile*'
_output_shapes
:���������*
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
�
lstm_1/while/Enter_3Enterlstm_1/Tile*'
_output_shapes
:���������*
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
�
lstm_1/while/Merge_2Mergelstm_1/while/Enter_2lstm_1/while/NextIteration_2*)
_output_shapes
:���������: *
T0*
N
�
lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*)
_output_shapes
:���������: *
T0*
N
�
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
�
lstm_1/while/SwitchSwitchlstm_1/while/Mergelstm_1/while/LoopCond*%
_class
loc:@lstm_1/while/Merge*
T0*
_output_shapes
: : 
�
lstm_1/while/Switch_1Switchlstm_1/while/Merge_1lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes

::
�
lstm_1/while/Switch_2Switchlstm_1/while/Merge_2lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_2*
T0*:
_output_shapes(
&:���������:���������
�
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:���������:���������
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
:���������
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*
T0*'
_output_shapes
:���������
�
$lstm_1/while/TensorArrayReadV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
�
&lstm_1/while/TensorArrayReadV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: **

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
�
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity&lstm_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*'
_output_shapes
:���������`
�
 lstm_1/while/strided_slice/stackConst^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:
�
"lstm_1/while/strided_slice/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:
�
"lstm_1/while/strided_slice/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/while/strided_sliceStridedSlicelstm_1/while/TensorArrayReadV3 lstm_1/while/strided_slice/stack"lstm_1/while/strided_slice/stack_1"lstm_1/while/strided_slice/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_1/stackConst^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:
�
$lstm_1/while/strided_slice_1/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    0   *
_output_shapes
:
�
$lstm_1/while/strided_slice_1/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/while/strided_slice_1StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_1/stack$lstm_1/while/strided_slice_1/stack_1$lstm_1/while/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_2/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    0   *
_output_shapes
:
�
$lstm_1/while/strided_slice_2/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    H   *
_output_shapes
:
�
$lstm_1/while/strided_slice_2/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/while/strided_slice_2StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_2/stack$lstm_1/while/strided_slice_2/stack_1$lstm_1/while/strided_slice_2/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_3/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    H   *
_output_shapes
:
�
$lstm_1/while/strided_slice_3/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:
�
$lstm_1/while/strided_slice_3/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/while/strided_slice_3StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_3/stack$lstm_1/while/strided_slice_3/stack_1$lstm_1/while/strided_slice_3/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
o
lstm_1/while/mul/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  �?*
_output_shapes
: 
v
lstm_1/while/mulMullstm_1/while/Identity_2lstm_1/while/mul/y*
T0*'
_output_shapes
:���������
�
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice_4*
_output_shapes

:*
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
�
lstm_1/while/MatMulMatMullstm_1/while/mullstm_1/while/MatMul/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
z
lstm_1/while/addAddlstm_1/while/strided_slicelstm_1/while/MatMul*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_1/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *��L>*
_output_shapes
: 
s
lstm_1/while/mul_1Mullstm_1/while/mul_1/xlstm_1/while/add*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
�
"lstm_1/while/clip_by_value/MinimumMinimumlstm_1/while/add_1lstm_1/while/Const_1*
T0*'
_output_shapes
:���������
�
lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_2/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  �?*
_output_shapes
: 
z
lstm_1/while/mul_2Mullstm_1/while/Identity_2lstm_1/while/mul_2/y*
T0*'
_output_shapes
:���������
�
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_5*
_output_shapes

:*
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
�
lstm_1/while/MatMul_1MatMullstm_1/while/mul_2lstm_1/while/MatMul_1/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/while/add_2Addlstm_1/while/strided_slice_1lstm_1/while/MatMul_1*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_3/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *��L>*
_output_shapes
: 
u
lstm_1/while/mul_3Mullstm_1/while/mul_3/xlstm_1/while/add_2*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
�
$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_3lstm_1/while/Const_3*
T0*'
_output_shapes
:���������
�
lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*
T0*'
_output_shapes
:���������
�
lstm_1/while/mul_4Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_3*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_5/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  �?*
_output_shapes
: 
z
lstm_1/while/mul_5Mullstm_1/while/Identity_2lstm_1/while/mul_5/y*
T0*'
_output_shapes
:���������
�
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_6*
_output_shapes

:*
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
�
lstm_1/while/MatMul_2MatMullstm_1/while/mul_5lstm_1/while/MatMul_2/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/while/add_4Addlstm_1/while/strided_slice_2lstm_1/while/MatMul_2*
T0*'
_output_shapes
:���������
_
lstm_1/while/TanhTanhlstm_1/while/add_4*
T0*'
_output_shapes
:���������
z
lstm_1/while/mul_6Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*
T0*'
_output_shapes
:���������
s
lstm_1/while/add_5Addlstm_1/while/mul_4lstm_1/while/mul_6*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_7/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  �?*
_output_shapes
: 
z
lstm_1/while/mul_7Mullstm_1/while/Identity_2lstm_1/while/mul_7/y*
T0*'
_output_shapes
:���������
�
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_7*
_output_shapes

:*
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
�
lstm_1/while/MatMul_3MatMullstm_1/while/mul_7lstm_1/while/MatMul_3/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/while/add_6Addlstm_1/while/strided_slice_3lstm_1/while/MatMul_3*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_8/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *��L>*
_output_shapes
: 
u
lstm_1/while/mul_8Mullstm_1/while/mul_8/xlstm_1/while/add_6*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
�
$lstm_1/while/clip_by_value_2/MinimumMinimumlstm_1/while/add_7lstm_1/while/Const_5*
T0*'
_output_shapes
:���������
�
lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:���������
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_5*
T0*'
_output_shapes
:���������
~
lstm_1/while/mul_9Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:���������
�
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/TensorArray*
is_constant(
�
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
�
lstm_1/while/NextIteration_1NextIteration0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
s
lstm_1/while/NextIteration_2NextIterationlstm_1/while/mul_9*
T0*'
_output_shapes
:���������
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/add_5*
T0*'
_output_shapes
:���������
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
:���������
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*
T0*'
_output_shapes
:���������
�
)lstm_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm_1/TensorArraylstm_1/while/Exit_1*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 
�
#lstm_1/TensorArrayStack/range/startConst*
dtype0*%
_class
loc:@lstm_1/TensorArray*
value	B : *
_output_shapes
: 
�
#lstm_1/TensorArrayStack/range/deltaConst*
dtype0*%
_class
loc:@lstm_1/TensorArray*
value	B :*
_output_shapes
: 
�
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*%
_class
loc:@lstm_1/TensorArray*

Tidx0*#
_output_shapes
:���������
�
+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_1*
dtype0*%
_class
loc:@lstm_1/TensorArray*4
_output_shapes"
 :������������������*$
element_shape:���������
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
�
lstm_1/TensorArrayReadV3TensorArrayReadV3lstm_1/TensorArray
lstm_1/sublstm_1/while/Exit_1*
dtype0*%
_class
loc:@lstm_1/TensorArray*'
_output_shapes
:���������
l
lstm_1/transpose_1/permConst*
dtype0*!
valueB"          *
_output_shapes
:
�
lstm_1/transpose_1	Transpose+lstm_1/TensorArrayStack/TensorArrayGatherV3lstm_1/transpose_1/perm*
Tperm0*
T0*4
_output_shapes"
 :������������������
e
lstm_1/Square_1Squarelstm_1/TensorArrayReadV3*
T0*'
_output_shapes
:���������
T
lstm_1/mul_11/xConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
h
lstm_1/mul_11Mullstm_1/mul_11/xlstm_1/Square_1*
T0*'
_output_shapes
:���������
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
valueB"      *
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
valueB
 *����*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
dtype0*
valueB
 *���>*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2�V*
seed���)*
T0*
_output_shapes

:
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:
�
dense_1/kernel
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:
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
�
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
�
dense_1/MatMulMatMullstm_1/TensorArrayReadV3dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������
U
lr/initial_valueConst*
dtype0*
valueB
 *
�#<*
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
�
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
�

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
�
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
�
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
:���������
i
dense_1_targetPlaceholder*
dtype0*
shape: *0
_output_shapes
:������������������
c
subSubdense_1/Tanhdense_1_target*
T0*0
_output_shapes
:������������������
P
SquareSquaresub*
T0*0
_output_shapes
:������������������
X
Mean/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
w
MeanMeanSquareMean/reduction_indices*#
_output_shapes
:���������*
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
:���������*
T0*
	keep_dims( *

Tidx0
X
mulMulMean_1dense_1_sample_weights*
T0*#
_output_shapes
:���������
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
:���������
S
CastCastNotEqual*

DstT0*

SrcT0
*#
_output_shapes
:���������
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
:���������
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
 *  �?*
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
 *  �?*
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
�
gradients/f_count_1Entergradients/f_count*
_output_shapes
: **

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 
�
gradients/MergeMergegradients/f_count_1gradients/NextIteration*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : *
N
�
gradients/SwitchSwitchgradients/Mergelstm_1/while/LoopCond*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 
�
gradients/Add/yConst^lstm_1/while/Identity*
dtype0*&
_class
loc:@lstm_1/while/Exit_1*
value	B :*
_output_shapes
: 
�
gradients/AddAddgradients/Switch:1gradients/Add/y*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
�+
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
�
gradients/b_count_1Entergradients/f_count_2*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 
�
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : *
N
�
gradients/GreaterEqual/EnterEntergradients/b_count*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant(
�
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
�
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 
�
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
�
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
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
_class

loc:@add_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Fill*gradients/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@add_1*
T0*
	keep_dims( *

Tidx0
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
_class

loc:@add_1*
T0*
_output_shapes
: *
Tshape0
�
gradients/add_1_grad/Sum_1Sumgradients/Fill,gradients/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@add_1*
T0*
	keep_dims( *

Tidx0
�
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
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
_class

loc:@add*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/add_1_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@add*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
_class

loc:@add*
T0*
_output_shapes
: *
Tshape0
�
gradients/add_grad/Sum_1Sumgradients/add_1_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@add*
T0*
	keep_dims( *

Tidx0
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
_class

loc:@add*
T0*
_output_shapes
: *
Tshape0
�
!gradients/lstm_1/add_9_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/add_9*
valueB *
_output_shapes
: 
�
#gradients/lstm_1/add_9_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/add_9*
valueB *
_output_shapes
: 
�
1gradients/lstm_1/add_9_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/lstm_1/add_9_grad/Shape#gradients/lstm_1/add_9_grad/Shape_1*
_class
loc:@lstm_1/add_9*
T0*2
_output_shapes 
:���������:���������
�
gradients/lstm_1/add_9_grad/SumSumgradients/add_1_grad/Reshape_11gradients/lstm_1/add_9_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@lstm_1/add_9*
T0*
	keep_dims( *

Tidx0
�
#gradients/lstm_1/add_9_grad/ReshapeReshapegradients/lstm_1/add_9_grad/Sum!gradients/lstm_1/add_9_grad/Shape*
_class
loc:@lstm_1/add_9*
T0*
_output_shapes
: *
Tshape0
�
!gradients/lstm_1/add_9_grad/Sum_1Sumgradients/add_1_grad/Reshape_13gradients/lstm_1/add_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class
loc:@lstm_1/add_9*
T0*
	keep_dims( *

Tidx0
�
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
�
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*
T0*2
_output_shapes 
:���������:���������
~
gradients/mul_1_grad/mulMulgradients/add_grad/ReshapeMean_3*
_class

loc:@mul_1*
T0*
_output_shapes
: 
�
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@mul_1*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
_class

loc:@mul_1*
T0*
_output_shapes
: *
Tshape0
�
gradients/mul_1_grad/mul_1Mulmul_1/xgradients/add_grad/Reshape*
_class

loc:@mul_1*
T0*
_output_shapes
: 
�
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@mul_1*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*
T0*
_output_shapes
: *
Tshape0
�
gradients/lstm_1/add_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/add*
valueB *
_output_shapes
: 
�
!gradients/lstm_1/add_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/add*
valueB *
_output_shapes
: 
�
/gradients/lstm_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/lstm_1/add_grad/Shape!gradients/lstm_1/add_grad/Shape_1*
_class
loc:@lstm_1/add*
T0*2
_output_shapes 
:���������:���������
�
gradients/lstm_1/add_grad/SumSumgradients/add_grad/Reshape_1/gradients/lstm_1/add_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@lstm_1/add*
T0*
	keep_dims( *

Tidx0
�
!gradients/lstm_1/add_grad/ReshapeReshapegradients/lstm_1/add_grad/Sumgradients/lstm_1/add_grad/Shape*
_class
loc:@lstm_1/add*
T0*
_output_shapes
: *
Tshape0
�
gradients/lstm_1/add_grad/Sum_1Sumgradients/add_grad/Reshape_11gradients/lstm_1/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class
loc:@lstm_1/add*
T0*
	keep_dims( *

Tidx0
�
#gradients/lstm_1/add_grad/Reshape_1Reshapegradients/lstm_1/add_grad/Sum_1!gradients/lstm_1/add_grad/Shape_1*
_class
loc:@lstm_1/add*
T0*
_output_shapes
: *
Tshape0
�
)gradients/lstm_1/Sum_2_grad/Reshape/shapeConst*
dtype0*
_class
loc:@lstm_1/Sum_2*
valueB"      *
_output_shapes
:
�
#gradients/lstm_1/Sum_2_grad/ReshapeReshape%gradients/lstm_1/add_9_grad/Reshape_1)gradients/lstm_1/Sum_2_grad/Reshape/shape*
_class
loc:@lstm_1/Sum_2*
T0*
_output_shapes

:*
Tshape0
�
!gradients/lstm_1/Sum_2_grad/ShapeShapelstm_1/mul_11*
out_type0*
T0*
_output_shapes
:*
_class
loc:@lstm_1/Sum_2
�
 gradients/lstm_1/Sum_2_grad/TileTile#gradients/lstm_1/Sum_2_grad/Reshape!gradients/lstm_1/Sum_2_grad/Shape*

Tmultiples0*
_class
loc:@lstm_1/Sum_2*
T0*'
_output_shapes
:���������
�
#gradients/Mean_3_grad/Reshape/shapeConst*
dtype0*
_class
loc:@Mean_3*
valueB:*
_output_shapes
:
�
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
�
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*

Tmultiples0*
_class
loc:@Mean_3*
T0*#
_output_shapes
:���������
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
�
gradients/Mean_3_grad/ConstConst*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
�
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
_output_shapes
: *
_class
loc:@Mean_3*
T0*
	keep_dims( *

Tidx0
�
gradients/Mean_3_grad/Const_1Const*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
�
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
�
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
�
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
�
gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

DstT0*
_class
loc:@Mean_3*

SrcT0*
_output_shapes
: 
�
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
_class
loc:@Mean_3*
T0*#
_output_shapes
:���������
�
'gradients/lstm_1/Sum_grad/Reshape/shapeConst*
dtype0*
_class
loc:@lstm_1/Sum*
valueB"      *
_output_shapes
:
�
!gradients/lstm_1/Sum_grad/ReshapeReshape#gradients/lstm_1/add_grad/Reshape_1'gradients/lstm_1/Sum_grad/Reshape/shape*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes

:*
Tshape0
�
(gradients/lstm_1/Sum_grad/Tile/multiplesConst*
dtype0*
_class
loc:@lstm_1/Sum*
valueB"   `   *
_output_shapes
:
�
gradients/lstm_1/Sum_grad/TileTile!gradients/lstm_1/Sum_grad/Reshape(gradients/lstm_1/Sum_grad/Tile/multiples*

Tmultiples0*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes

:`
�
"gradients/lstm_1/mul_11_grad/ShapeConst*
dtype0* 
_class
loc:@lstm_1/mul_11*
valueB *
_output_shapes
: 
�
$gradients/lstm_1/mul_11_grad/Shape_1Shapelstm_1/Square_1*
out_type0*
T0*
_output_shapes
:* 
_class
loc:@lstm_1/mul_11
�
2gradients/lstm_1/mul_11_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/lstm_1/mul_11_grad/Shape$gradients/lstm_1/mul_11_grad/Shape_1* 
_class
loc:@lstm_1/mul_11*
T0*2
_output_shapes 
:���������:���������
�
 gradients/lstm_1/mul_11_grad/mulMul gradients/lstm_1/Sum_2_grad/Tilelstm_1/Square_1* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:���������
�
 gradients/lstm_1/mul_11_grad/SumSum gradients/lstm_1/mul_11_grad/mul2gradients/lstm_1/mul_11_grad/BroadcastGradientArgs*
_output_shapes
:* 
_class
loc:@lstm_1/mul_11*
T0*
	keep_dims( *

Tidx0
�
$gradients/lstm_1/mul_11_grad/ReshapeReshape gradients/lstm_1/mul_11_grad/Sum"gradients/lstm_1/mul_11_grad/Shape* 
_class
loc:@lstm_1/mul_11*
T0*
_output_shapes
: *
Tshape0
�
"gradients/lstm_1/mul_11_grad/mul_1Mullstm_1/mul_11/x gradients/lstm_1/Sum_2_grad/Tile* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:���������
�
"gradients/lstm_1/mul_11_grad/Sum_1Sum"gradients/lstm_1/mul_11_grad/mul_14gradients/lstm_1/mul_11_grad/BroadcastGradientArgs:1*
_output_shapes
:* 
_class
loc:@lstm_1/mul_11*
T0*
	keep_dims( *

Tidx0
�
&gradients/lstm_1/mul_11_grad/Reshape_1Reshape"gradients/lstm_1/mul_11_grad/Sum_1$gradients/lstm_1/mul_11_grad/Shape_1* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:���������*
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
�
(gradients/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_grad/Shapegradients/div_grad/Shape_1*
_class

loc:@div*
T0*2
_output_shapes 
:���������:���������
�
gradients/div_grad/RealDivRealDivgradients/Mean_3_grad/truedivMean_2*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/SumSumgradients/div_grad/RealDiv(gradients/div_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@div*
T0*
	keep_dims( *

Tidx0
�
gradients/div_grad/ReshapeReshapegradients/div_grad/Sumgradients/div_grad/Shape*
_class

loc:@div*
T0*#
_output_shapes
:���������*
Tshape0
h
gradients/div_grad/NegNegmul*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/RealDiv_1RealDivgradients/div_grad/NegMean_2*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/RealDiv_2RealDivgradients/div_grad/RealDiv_1Mean_2*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/mulMulgradients/Mean_3_grad/truedivgradients/div_grad/RealDiv_2*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/Sum_1Sumgradients/div_grad/mul*gradients/div_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@div*
T0*
	keep_dims( *

Tidx0
�
gradients/div_grad/Reshape_1Reshapegradients/div_grad/Sum_1gradients/div_grad/Shape_1*
_class

loc:@div*
T0*
_output_shapes
: *
Tshape0
�
gradients/lstm_1/mul_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/mul*
valueB *
_output_shapes
: 
�
!gradients/lstm_1/mul_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/mul*
valueB"   `   *
_output_shapes
:
�
/gradients/lstm_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/lstm_1/mul_grad/Shape!gradients/lstm_1/mul_grad/Shape_1*
_class
loc:@lstm_1/mul*
T0*2
_output_shapes 
:���������:���������
�
gradients/lstm_1/mul_grad/mulMulgradients/lstm_1/Sum_grad/Tilelstm_1/Square*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:`
�
gradients/lstm_1/mul_grad/SumSumgradients/lstm_1/mul_grad/mul/gradients/lstm_1/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
_class
loc:@lstm_1/mul*
T0*
	keep_dims( *

Tidx0
�
!gradients/lstm_1/mul_grad/ReshapeReshapegradients/lstm_1/mul_grad/Sumgradients/lstm_1/mul_grad/Shape*
_class
loc:@lstm_1/mul*
T0*
_output_shapes
: *
Tshape0
�
gradients/lstm_1/mul_grad/mul_1Mullstm_1/mul/xgradients/lstm_1/Sum_grad/Tile*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:`
�
gradients/lstm_1/mul_grad/Sum_1Sumgradients/lstm_1/mul_grad/mul_11gradients/lstm_1/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class
loc:@lstm_1/mul*
T0*
	keep_dims( *

Tidx0
�
#gradients/lstm_1/mul_grad/Reshape_1Reshapegradients/lstm_1/mul_grad/Sum_1!gradients/lstm_1/mul_grad/Shape_1*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:`*
Tshape0
�
$gradients/lstm_1/Square_1_grad/mul/xConst'^gradients/lstm_1/mul_11_grad/Reshape_1*
dtype0*"
_class
loc:@lstm_1/Square_1*
valueB
 *   @*
_output_shapes
: 
�
"gradients/lstm_1/Square_1_grad/mulMul$gradients/lstm_1/Square_1_grad/mul/xlstm_1/TensorArrayReadV3*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������
�
$gradients/lstm_1/Square_1_grad/mul_1Mul&gradients/lstm_1/mul_11_grad/Reshape_1"gradients/lstm_1/Square_1_grad/mul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������
v
gradients/mul_grad/ShapeShapeMean_1*
out_type0*
T0*
_output_shapes
:*
_class

loc:@mul
�
gradients/mul_grad/Shape_1Shapedense_1_sample_weights*
out_type0*
T0*
_output_shapes
:*
_class

loc:@mul
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
_class

loc:@mul*
T0*2
_output_shapes 
:���������:���������
�
gradients/mul_grad/mulMulgradients/div_grad/Reshapedense_1_sample_weights*
_class

loc:@mul*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@mul*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
_class

loc:@mul*
T0*#
_output_shapes
:���������*
Tshape0
�
gradients/mul_grad/mul_1MulMean_1gradients/div_grad/Reshape*
_class

loc:@mul*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*
_class

loc:@mul*
T0*
	keep_dims( *

Tidx0
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
_class

loc:@mul*
T0*#
_output_shapes
:���������*
Tshape0
�
"gradients/lstm_1/Square_grad/mul/xConst$^gradients/lstm_1/mul_grad/Reshape_1*
dtype0* 
_class
loc:@lstm_1/Square*
valueB
 *   @*
_output_shapes
: 
�
 gradients/lstm_1/Square_grad/mulMul"gradients/lstm_1/Square_grad/mul/xlstm_1/kernel/read* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:`
�
"gradients/lstm_1/Square_grad/mul_1Mul#gradients/lstm_1/mul_grad/Reshape_1 gradients/lstm_1/Square_grad/mul* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:`
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
�
gradients/Mean_1_grad/addAddMean_1/reduction_indicesgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
gradients/Mean_1_grad/modFloorModgradients/Mean_1_grad/addgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
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
�
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
�
gradients/Mean_1_grad/FillFillgradients/Mean_1_grad/Shape_1 gradients/Mean_1_grad/Fill/value*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
#gradients/Mean_1_grad/DynamicStitchDynamicStitchgradients/Mean_1_grad/rangegradients/Mean_1_grad/modgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Fill*
_class
loc:@Mean_1*
T0*#
_output_shapes
:���������*
N
|
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
�
gradients/Mean_1_grad/MaximumMaximum#gradients/Mean_1_grad/DynamicStitchgradients/Mean_1_grad/Maximum/y*
_class
loc:@Mean_1*
T0*#
_output_shapes
:���������
�
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Maximum*
_class
loc:@Mean_1*
T0*#
_output_shapes
:���������
�
gradients/Mean_1_grad/ReshapeReshapegradients/mul_grad/Reshape#gradients/Mean_1_grad/DynamicStitch*
_class
loc:@Mean_1*
T0*
_output_shapes
:*
Tshape0
�
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
�
gradients/Mean_1_grad/ConstConst*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
�
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const*
_output_shapes
: *
_class
loc:@Mean_1*
T0*
	keep_dims( *

Tidx0
�
gradients/Mean_1_grad/Const_1Const*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
�
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
�
gradients/Mean_1_grad/Maximum_1Maximumgradients/Mean_1_grad/Prod_1!gradients/Mean_1_grad/Maximum_1/y*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
 gradients/Mean_1_grad/floordiv_1FloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum_1*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
gradients/Mean_1_grad/CastCast gradients/Mean_1_grad/floordiv_1*

DstT0*
_class
loc:@Mean_1*

SrcT0*
_output_shapes
: 
�
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
_class
loc:@Mean_1*
T0*#
_output_shapes
:���������
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
�
gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 
�
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
�
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
�
gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
_class
	loc:@Mean*
T0*
_output_shapes
: 
�
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
_class
	loc:@Mean*
T0*#
_output_shapes
:���������*
N
x
gradients/Mean_grad/Maximum/yConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
_class
	loc:@Mean*
T0*#
_output_shapes
:���������
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*
T0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Mean_1_grad/truediv!gradients/Mean_grad/DynamicStitch*
_class
	loc:@Mean*
T0*
_output_shapes
:*
Tshape0
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
_class
	loc:@Mean*
T0*0
_output_shapes
:������������������
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
�
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
�
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
�
gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_class
	loc:@Mean*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
T0*
_output_shapes
: 
�
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
_class
	loc:@Mean*

SrcT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_class
	loc:@Mean*
T0*0
_output_shapes
:������������������
�
gradients/Square_grad/mul/xConst^gradients/Mean_grad/truediv*
dtype0*
_class
loc:@Square*
valueB
 *   @*
_output_shapes
: 
�
gradients/Square_grad/mulMulgradients/Square_grad/mul/xsub*
_class
loc:@Square*
T0*0
_output_shapes
:������������������
�
gradients/Square_grad/mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/mul*
_class
loc:@Square*
T0*0
_output_shapes
:������������������
|
gradients/sub_grad/ShapeShapedense_1/Tanh*
out_type0*
T0*
_output_shapes
:*
_class

loc:@sub
�
gradients/sub_grad/Shape_1Shapedense_1_target*
out_type0*
T0*
_output_shapes
:*
_class

loc:@sub
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
_class

loc:@sub*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSumgradients/Square_grad/mul_1(gradients/sub_grad/BroadcastGradientArgs*
_output_shapes
:*
_class

loc:@sub*
T0*
	keep_dims( *

Tidx0
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
_class

loc:@sub*
T0*'
_output_shapes
:���������*
Tshape0
�
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
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
_class

loc:@sub*
T0*0
_output_shapes
:������������������*
Tshape0
�
$gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhgradients/sub_grad/Reshape*
_class
loc:@dense_1/Tanh*
T0*'
_output_shapes
:���������
�
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_1/Tanh_grad/TanhGrad*"
_class
loc:@dense_1/BiasAdd*
T0*
_output_shapes
:*
data_formatNHWC
�
$gradients/dense_1/MatMul_grad/MatMulMatMul$gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
transpose_b(*
transpose_a( *!
_class
loc:@dense_1/MatMul*
T0*'
_output_shapes
:���������
�
&gradients/dense_1/MatMul_grad/MatMul_1MatMullstm_1/TensorArrayReadV3$gradients/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_1/MatMul*
T0*
_output_shapes

:
�
gradients/AddNAddN$gradients/lstm_1/Square_1_grad/mul_1$gradients/dense_1/MatMul_grad/MatMul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������*
N
�
Igradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArraylstm_1/while/Exit_1*
source	gradients*%
_class
loc:@lstm_1/TensorArray*
_output_shapes

::
�
Egradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitylstm_1/while/Exit_1J^gradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
:
�
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
:���������
j
gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_3*
T0*'
_output_shapes
:���������
�
)gradients/lstm_1/while/Exit_1_grad/b_exitEnterKgradients/lstm_1/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 
�
)gradients/lstm_1/while/Exit_2_grad/b_exitEntergradients/zeros_like*'
_output_shapes
:���������*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_2*
is_constant( 
�
)gradients/lstm_1/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*'
_output_shapes
:���������*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_3*
is_constant( 
�
-gradients/lstm_1/while/Switch_1_grad/b_switchMerge)gradients/lstm_1/while/Exit_1_grad/b_exit4gradients/lstm_1/while/Switch_1_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: : *
N
�
-gradients/lstm_1/while/Switch_2_grad/b_switchMerge)gradients/lstm_1/while/Exit_2_grad/b_exit4gradients/lstm_1/while/Switch_2_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_2*
T0*)
_output_shapes
:���������: *
N
�
-gradients/lstm_1/while/Switch_3_grad/b_switchMerge)gradients/lstm_1/while/Exit_3_grad/b_exit4gradients/lstm_1/while/Switch_3_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_3*
T0*)
_output_shapes
:���������: *
N
�
*gradients/lstm_1/while/Merge_1_grad/SwitchSwitch-gradients/lstm_1/while/Switch_1_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: : 
�
*gradients/lstm_1/while/Merge_2_grad/SwitchSwitch-gradients/lstm_1/while/Switch_2_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_2*
T0*:
_output_shapes(
&:���������:���������
�
*gradients/lstm_1/while/Merge_3_grad/SwitchSwitch-gradients/lstm_1/while/Switch_3_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:���������:���������
�
(gradients/lstm_1/while/Enter_1_grad/ExitExit*gradients/lstm_1/while/Merge_1_grad/Switch*'
_class
loc:@lstm_1/while/Enter_1*
T0*
_output_shapes
: 
�
ggradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/TensorArray*
is_constant(
�
agradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3ggradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter,gradients/lstm_1/while/Merge_1_grad/Switch:1*
source	gradients*%
_class
loc:@lstm_1/TensorArray*
_output_shapes

::
�
]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity,gradients/lstm_1/while/Merge_1_grad/Switch:1b^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
: 
�
Wgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
	elem_type0*
_output_shapes
:
�
Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnterRefEnterWgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
is_constant(
�
[gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPush	StackPushZgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnterlstm_1/while/Identity^gradients/Add*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
T0*
swap_memory(*
_output_shapes
:
�
cgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop/RefEnterRefEnterWgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
is_constant(
�
Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopStackPopcgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
	elem_type0*
_output_shapes
: 
�+
Xgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger[^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopA^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_9_grad/mul/StackPop1^gradients/lstm_1/while/mul_9_grad/mul_1/StackPopB^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopD^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1K^gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopG^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopI^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1S^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1A^gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_4_grad/mul/StackPop1^gradients/lstm_1/while/mul_4_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_6_grad/mul/StackPop1^gradients/lstm_1/while/mul_6_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_8_grad/mul/StackPop1^gradients/lstm_1/while/mul_8_grad/mul_1/StackPopB^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopD^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1K^gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop@^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopB^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1I^gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1G^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopI^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1S^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopE^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopG^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1Q^gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1F^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopA^gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopF^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPopA^gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_7_grad/mul/StackPop1^gradients/lstm_1/while/mul_7_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_3_grad/mul/StackPop1^gradients/lstm_1/while/mul_3_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_1_grad/mul/StackPop1^gradients/lstm_1/while/mul_1_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_5_grad/mul/StackPopA^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1?^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1F^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPopD^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopF^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1F^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2F^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_35^gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPopA^gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_2_grad/mul/StackPop?^gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop-^gradients/lstm_1/while/mul_grad/mul/StackPop*%
_class
loc:@lstm_1/TensorArray
�
Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3agradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*%
_class
loc:@lstm_1/TensorArray*'
_output_shapes
:���������
�
gradients/AddN_1AddN,gradients/lstm_1/while/Merge_2_grad/Switch:1Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:���������*
N
�
'gradients/lstm_1/while/mul_9_grad/ShapeShapelstm_1/while/clip_by_value_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_9
�
)gradients/lstm_1/while/mul_9_grad/Shape_1Shapelstm_1/while/Tanh_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_9
�
=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
�
Agradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_9_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
�
@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
�
Cgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_9_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
�
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_9*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_9_grad/mul/f_accStack*

stack_name *?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_9_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_9_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
is_constant(
�
/gradients/lstm_1/while/mul_9_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_9_grad/mul/RefEnterlstm_1/while/Tanh_1^gradients/Add*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_9_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_9_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
is_constant(
�
.gradients/lstm_1/while/mul_9_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_9_grad/mul/StackPop/RefEnter^gradients/Sub*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_9_grad/mulMulgradients/AddN_1.gradients/lstm_1/while/mul_9_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_9_grad/SumSum%gradients/lstm_1/while/mul_9_grad/mul7gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_9*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_9_grad/ReshapeReshape%gradients/lstm_1/while/mul_9_grad/Sum@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������*
Tshape0
�
-gradients/lstm_1/while/mul_9_grad/mul_1/f_accStack*

stack_name *H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_9_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_9_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
is_constant(
�
1gradients/lstm_1/while/mul_9_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_9_grad/mul_1/RefEnterlstm_1/while/clip_by_value_2^gradients/Add*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_9_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_9_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
is_constant(
�
0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_9_grad/mul_1/StackPop/RefEnter^gradients/Sub*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
	elem_type0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_9_grad/mul_1Mul0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopgradients/AddN_1*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_9_grad/Sum_1Sum'gradients/lstm_1/while/mul_9_grad/mul_19gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_9*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_9_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_9_grad/Sum_1Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������*
Tshape0
�
1gradients/lstm_1/while/clip_by_value_2_grad/ShapeShape$lstm_1/while/clip_by_value_2/Minimum*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
�
3gradients/lstm_1/while/clip_by_value_2_grad/Shape_1Const^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB *
_output_shapes
: 
�
3gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Shape)gradients/lstm_1/while/mul_9_grad/Reshape*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_2
�
7gradients/lstm_1/while/clip_by_value_2_grad/zeros/ConstConst^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
_output_shapes
: 
�
1gradients/lstm_1/while/clip_by_value_2_grad/zerosFill3gradients/lstm_1/while/clip_by_value_2_grad/Shape_27gradients/lstm_1/while/clip_by_value_2_grad/zeros/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_accStack*

stack_name *Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Bgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush	StackPushAgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter$lstm_1/while/clip_by_value_2/Minimum^gradients/Add*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Agradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopStackPopJgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*'
_output_shapes
:���������
�
@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1Stack*

stack_name *J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
�
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1RefEnter@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
is_constant(
�
Dgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush_1	StackPushCgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1lstm_1/while/Const_4^gradients/Add*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
T0*
swap_memory(*
_output_shapes
:
�
Lgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1/RefEnterRefEnter@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
is_constant(
�
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1StackPopLgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
: 
�
8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqualGreaterEqualAgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopCgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
Ggradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
is_constant(
�
Kgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPush	StackPushJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnter1gradients/lstm_1/while/clip_by_value_2_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
swap_memory(*
_output_shapes
:
�
Sgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
is_constant(
�
Jgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopStackPopSgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop3gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*2
_output_shapes 
:���������:���������
�
2gradients/lstm_1/while/clip_by_value_2_grad/SelectSelect8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:���������
�
4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
/gradients/lstm_1/while/clip_by_value_2_grad/SumSum2gradients/lstm_1/while/clip_by_value_2_grad/SelectAgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
	keep_dims( *

Tidx0
�
3gradients/lstm_1/while/clip_by_value_2_grad/ReshapeReshape/gradients/lstm_1/while/clip_by_value_2_grad/SumJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������*
Tshape0
�
1gradients/lstm_1/while/clip_by_value_2_grad/Sum_1Sum4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Cgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
	keep_dims( *

Tidx0
�
5gradients/lstm_1/while/clip_by_value_2_grad/Reshape_1Reshape1gradients/lstm_1/while/clip_by_value_2_grad/Sum_13gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
_output_shapes
: *
Tshape0
�
+gradients/lstm_1/while/Tanh_1_grad/TanhGradTanhGrad.gradients/lstm_1/while/mul_9_grad/mul/StackPop+gradients/lstm_1/while/mul_9_grad/Reshape_1*&
_class
loc:@lstm_1/while/Tanh_1*
T0*'
_output_shapes
:���������
�
4gradients/lstm_1/while/Switch_1_grad_1/NextIterationNextIteration,gradients/lstm_1/while/Merge_1_grad/Switch:1*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: 
�
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ShapeShapelstm_1/while/add_7*
out_type0*
T0*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
�
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB *
_output_shapes
: 
�
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2Shape3gradients/lstm_1/while/clip_by_value_2_grad/Reshape*
out_type0*
T0*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum
�
?gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *    *
_output_shapes
: 
�
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosFill;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2?gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
Cgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_accStack*

stack_name *P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Fgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Ggradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush	StackPushFgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_7^gradients/Add*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Ogradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Fgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopStackPopOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*'
_output_shapes
:���������
�
Egradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1RefEnterEgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush_1	StackPushHgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_5^gradients/Add*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Qgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1StackPopQgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
: 
�
=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual	LessEqualFgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopHgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
Ogradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Rgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Sgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnter9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape^gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
[gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Rgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopStackPop[gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*2
_output_shapes 
:���������:���������
�
:gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectSelect=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:���������
�
<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
7gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumSum:gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectIgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
	keep_dims( *

Tidx0
�
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeReshape7gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������*
Tshape0
�
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1Sum<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Kgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
	keep_dims( *

Tidx0
�
=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape_1Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
_output_shapes
: *
Tshape0
�
gradients/AddN_2AddN,gradients/lstm_1/while/Merge_3_grad/Switch:1+gradients/lstm_1/while/Tanh_1_grad/TanhGrad*'
_class
loc:@lstm_1/while/Merge_3*
T0*'
_output_shapes
:���������*
N
�
'gradients/lstm_1/while/add_5_grad/ShapeShapelstm_1/while/mul_4*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_5
�
)gradients/lstm_1/while/add_5_grad/Shape_1Shapelstm_1/while/mul_6*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_5
�
=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
�
Agradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_5_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_5*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
�
@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
�
Cgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_5_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_5*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
�
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_5*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_5_grad/SumSumgradients/AddN_27gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_5*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/add_5_grad/ReshapeReshape%gradients/lstm_1/while/add_5_grad/Sum@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_5*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/add_5_grad/Sum_1Sumgradients/AddN_29gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_5*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/add_5_grad/Reshape_1Reshape'gradients/lstm_1/while/add_5_grad/Sum_1Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_5*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/add_7_grad/ShapeShapelstm_1/while/mul_8*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_7
�
)gradients/lstm_1/while/add_7_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_7*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_7*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_7*
is_constant(
�
Agradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_7_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_7*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_7*
is_constant(
�
@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_7*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_7_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_7_grad/Shape_1*%
_class
loc:@lstm_1/while/add_7*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_7_grad/SumSum;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape7gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_7*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/add_7_grad/ReshapeReshape%gradients/lstm_1/while/add_7_grad/Sum@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_7*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/add_7_grad/Sum_1Sum;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape9gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_7*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/add_7_grad/Reshape_1Reshape'gradients/lstm_1/while/add_7_grad/Sum_1)gradients/lstm_1/while/add_7_grad/Shape_1*%
_class
loc:@lstm_1/while/add_7*
T0*
_output_shapes
: *
Tshape0
�
'gradients/lstm_1/while/mul_4_grad/ShapeShapelstm_1/while/clip_by_value_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_4
�
)gradients/lstm_1/while/mul_4_grad/Shape_1Shapelstm_1/while/Identity_3*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_4
�
=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
�
Agradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_4_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
�
@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
�
Cgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_4_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
�
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_4*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_4_grad/mul/f_accStack*

stack_name *C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_4_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_4_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
is_constant(
�
/gradients/lstm_1/while/mul_4_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_4_grad/mul/RefEnterlstm_1/while/Identity_3^gradients/Add*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_4_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_4_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
is_constant(
�
.gradients/lstm_1/while/mul_4_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_4_grad/mul/StackPop/RefEnter^gradients/Sub*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_4_grad/mulMul)gradients/lstm_1/while/add_5_grad/Reshape.gradients/lstm_1/while/mul_4_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_4_grad/SumSum%gradients/lstm_1/while/mul_4_grad/mul7gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_4*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_4_grad/ReshapeReshape%gradients/lstm_1/while/mul_4_grad/Sum@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������*
Tshape0
�
-gradients/lstm_1/while/mul_4_grad/mul_1/f_accStack*

stack_name *H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_4_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_4_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
is_constant(
�
1gradients/lstm_1/while/mul_4_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_4_grad/mul_1/RefEnterlstm_1/while/clip_by_value_1^gradients/Add*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_4_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_4_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
is_constant(
�
0gradients/lstm_1/while/mul_4_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_4_grad/mul_1/StackPop/RefEnter^gradients/Sub*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
	elem_type0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_4_grad/mul_1Mul0gradients/lstm_1/while/mul_4_grad/mul_1/StackPop)gradients/lstm_1/while/add_5_grad/Reshape*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_4_grad/Sum_1Sum'gradients/lstm_1/while/mul_4_grad/mul_19gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_4*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_4_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_4_grad/Sum_1Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/mul_6_grad/ShapeShapelstm_1/while/clip_by_value*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_6
�
)gradients/lstm_1/while/mul_6_grad/Shape_1Shapelstm_1/while/Tanh*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_6
�
=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
�
Agradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_6_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
�
@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
�
Cgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_6_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
�
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_6*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_6_grad/mul/f_accStack*

stack_name *=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_6_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_6_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
is_constant(
�
/gradients/lstm_1/while/mul_6_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_6_grad/mul/RefEnterlstm_1/while/Tanh^gradients/Add*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_6_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_6_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
is_constant(
�
.gradients/lstm_1/while/mul_6_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_6_grad/mul/StackPop/RefEnter^gradients/Sub*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_6_grad/mulMul+gradients/lstm_1/while/add_5_grad/Reshape_1.gradients/lstm_1/while/mul_6_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_6_grad/SumSum%gradients/lstm_1/while/mul_6_grad/mul7gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_6*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_6_grad/ReshapeReshape%gradients/lstm_1/while/mul_6_grad/Sum@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������*
Tshape0
�
-gradients/lstm_1/while/mul_6_grad/mul_1/f_accStack*

stack_name *F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_6_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_6_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
is_constant(
�
1gradients/lstm_1/while/mul_6_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_6_grad/mul_1/RefEnterlstm_1/while/clip_by_value^gradients/Add*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_6_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_6_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
is_constant(
�
0gradients/lstm_1/while/mul_6_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_6_grad/mul_1/StackPop/RefEnter^gradients/Sub*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
	elem_type0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_6_grad/mul_1Mul0gradients/lstm_1/while/mul_6_grad/mul_1/StackPop+gradients/lstm_1/while/add_5_grad/Reshape_1*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_6_grad/Sum_1Sum'gradients/lstm_1/while/mul_6_grad/mul_19gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_6*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_6_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_6_grad/Sum_1Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/mul_8_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_8*
valueB *
_output_shapes
: 
�
)gradients/lstm_1/while/mul_8_grad/Shape_1Shapelstm_1/while/add_6*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_8
�
=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_8*
is_constant(
�
Agradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_8_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_8*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_8*
is_constant(
�
@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_8_grad/Shape@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_8_grad/mul/f_accStack*

stack_name *>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_8_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_8_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
is_constant(
�
/gradients/lstm_1/while/mul_8_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_8_grad/mul/RefEnterlstm_1/while/add_6^gradients/Add*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_8_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_8_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
is_constant(
�
.gradients/lstm_1/while/mul_8_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_8_grad/mul/StackPop/RefEnter^gradients/Sub*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_8_grad/mulMul)gradients/lstm_1/while/add_7_grad/Reshape.gradients/lstm_1/while/mul_8_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_8_grad/SumSum%gradients/lstm_1/while/mul_8_grad/mul7gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_8*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_8_grad/ReshapeReshape%gradients/lstm_1/while/mul_8_grad/Sum'gradients/lstm_1/while/mul_8_grad/Shape*%
_class
loc:@lstm_1/while/mul_8*
T0*
_output_shapes
: *
Tshape0
�
-gradients/lstm_1/while/mul_8_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_8_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_8_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
is_constant(
�
1gradients/lstm_1/while/mul_8_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_8_grad/mul_1/RefEnterlstm_1/while/mul_8/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_8_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_8_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
is_constant(
�
0gradients/lstm_1/while/mul_8_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_8_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
	elem_type0*
_output_shapes
: 
�
'gradients/lstm_1/while/mul_8_grad/mul_1Mul0gradients/lstm_1/while/mul_8_grad/mul_1/StackPop)gradients/lstm_1/while/add_7_grad/Reshape*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_8_grad/Sum_1Sum'gradients/lstm_1/while/mul_8_grad/mul_19gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_8*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_8_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_8_grad/Sum_1@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:���������*
Tshape0
�
1gradients/lstm_1/while/clip_by_value_1_grad/ShapeShape$lstm_1/while/clip_by_value_1/Minimum*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
�
3gradients/lstm_1/while/clip_by_value_1_grad/Shape_1Const^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB *
_output_shapes
: 
�
3gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Shape)gradients/lstm_1/while/mul_4_grad/Reshape*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_1
�
7gradients/lstm_1/while/clip_by_value_1_grad/zeros/ConstConst^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB
 *    *
_output_shapes
: 
�
1gradients/lstm_1/while/clip_by_value_1_grad/zerosFill3gradients/lstm_1/while/clip_by_value_1_grad/Shape_27gradients/lstm_1/while/clip_by_value_1_grad/zeros/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_accStack*

stack_name *Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Bgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush	StackPushAgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter$lstm_1/while/clip_by_value_1/Minimum^gradients/Add*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Agradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopStackPopJgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*'
_output_shapes
:���������
�
@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1Stack*

stack_name *J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
�
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1RefEnter@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
is_constant(
�
Dgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush_1	StackPushCgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1lstm_1/while/Const_2^gradients/Add*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
T0*
swap_memory(*
_output_shapes
:
�
Lgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1/RefEnterRefEnter@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
is_constant(
�
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1StackPopLgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
: 
�
8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqualGreaterEqualAgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopCgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
Ggradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
is_constant(
�
Kgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPush	StackPushJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnter1gradients/lstm_1/while/clip_by_value_1_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
swap_memory(*
_output_shapes
:
�
Sgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
is_constant(
�
Jgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopStackPopSgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop3gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*2
_output_shapes 
:���������:���������
�
2gradients/lstm_1/while/clip_by_value_1_grad/SelectSelect8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:���������
�
4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
/gradients/lstm_1/while/clip_by_value_1_grad/SumSum2gradients/lstm_1/while/clip_by_value_1_grad/SelectAgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
	keep_dims( *

Tidx0
�
3gradients/lstm_1/while/clip_by_value_1_grad/ReshapeReshape/gradients/lstm_1/while/clip_by_value_1_grad/SumJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������*
Tshape0
�
1gradients/lstm_1/while/clip_by_value_1_grad/Sum_1Sum4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Cgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
	keep_dims( *

Tidx0
�
5gradients/lstm_1/while/clip_by_value_1_grad/Reshape_1Reshape1gradients/lstm_1/while/clip_by_value_1_grad/Sum_13gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
_output_shapes
: *
Tshape0
�
/gradients/lstm_1/while/clip_by_value_grad/ShapeShape"lstm_1/while/clip_by_value/Minimum*
out_type0*
T0*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/clip_by_value
�
1gradients/lstm_1/while/clip_by_value_grad/Shape_1Const^gradients/Sub*
dtype0*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB *
_output_shapes
: 
�
1gradients/lstm_1/while/clip_by_value_grad/Shape_2Shape)gradients/lstm_1/while/mul_6_grad/Reshape*
out_type0*
T0*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/clip_by_value
�
5gradients/lstm_1/while/clip_by_value_grad/zeros/ConstConst^gradients/Sub*
dtype0*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    *
_output_shapes
: 
�
/gradients/lstm_1/while/clip_by_value_grad/zerosFill1gradients/lstm_1/while/clip_by_value_grad/Shape_25gradients/lstm_1/while/clip_by_value_grad/zeros/Const*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_accStack*

stack_name *V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnterRefEnter<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
@gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush	StackPush?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter"lstm_1/while/clip_by_value/Minimum^gradients/Add*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Hgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop/RefEnterRefEnter<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopStackPopHgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1Stack*

stack_name *F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1RefEnter>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
is_constant(
�
Bgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush_1	StackPushAgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1lstm_1/while/Const^gradients/Add*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
T0*
swap_memory(*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
is_constant(
�
Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1StackPopJgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
: 
�
6gradients/lstm_1/while/clip_by_value_grad/GreaterEqualGreaterEqual?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopAgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
Egradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_accStack*

stack_name *-
_class#
!loc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
�
Hgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
is_constant(
�
Igradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPush	StackPushHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnter/gradients/lstm_1/while/clip_by_value_grad/Shape^gradients/Add*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
swap_memory(*
_output_shapes
:
�
Qgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
is_constant(
�
Hgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopStackPopQgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop1gradients/lstm_1/while/clip_by_value_grad/Shape_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*2
_output_shapes 
:���������:���������
�
0gradients/lstm_1/while/clip_by_value_grad/SelectSelect6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
4gradients/lstm_1/while/clip_by_value_grad/LogicalNot
LogicalNot6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:���������
�
2gradients/lstm_1/while/clip_by_value_grad/Select_1Select4gradients/lstm_1/while/clip_by_value_grad/LogicalNot)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/clip_by_value_grad/SumSum0gradients/lstm_1/while/clip_by_value_grad/Select?gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
	keep_dims( *

Tidx0
�
1gradients/lstm_1/while/clip_by_value_grad/ReshapeReshape-gradients/lstm_1/while/clip_by_value_grad/SumHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������*
Tshape0
�
/gradients/lstm_1/while/clip_by_value_grad/Sum_1Sum2gradients/lstm_1/while/clip_by_value_grad/Select_1Agradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs:1*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
	keep_dims( *

Tidx0
�
3gradients/lstm_1/while/clip_by_value_grad/Reshape_1Reshape/gradients/lstm_1/while/clip_by_value_grad/Sum_11gradients/lstm_1/while/clip_by_value_grad/Shape_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
_output_shapes
: *
Tshape0
�
)gradients/lstm_1/while/Tanh_grad/TanhGradTanhGrad.gradients/lstm_1/while/mul_6_grad/mul/StackPop+gradients/lstm_1/while/mul_6_grad/Reshape_1*$
_class
loc:@lstm_1/while/Tanh*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_6_grad/ShapeShapelstm_1/while/strided_slice_3*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_6
�
)gradients/lstm_1/while/add_6_grad/Shape_1Shapelstm_1/while/MatMul_3*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_6
�
=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
�
Agradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_6_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_6*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
�
@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
�
Cgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_6_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_6*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
�
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_6*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_6_grad/SumSum+gradients/lstm_1/while/mul_8_grad/Reshape_17gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_6*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/add_6_grad/ReshapeReshape%gradients/lstm_1/while/add_6_grad/Sum@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_6*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/add_6_grad/Sum_1Sum+gradients/lstm_1/while/mul_8_grad/Reshape_19gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_6*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/add_6_grad/Reshape_1Reshape'gradients/lstm_1/while/add_6_grad/Sum_1Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_6*
T0*'
_output_shapes
:���������*
Tshape0
�
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ShapeShapelstm_1/while/add_3*
out_type0*
T0*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
�
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB *
_output_shapes
: 
�
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2Shape3gradients/lstm_1/while/clip_by_value_1_grad/Reshape*
out_type0*
T0*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum
�
?gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB
 *    *
_output_shapes
: 
�
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosFill;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2?gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
Cgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_accStack*

stack_name *P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Fgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Ggradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush	StackPushFgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_3^gradients/Add*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Ogradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Fgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopStackPopOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*'
_output_shapes
:���������
�
Egradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1RefEnterEgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush_1	StackPushHgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_3^gradients/Add*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Qgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1StackPopQgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
: 
�
=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual	LessEqualFgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopHgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
Ogradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Rgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Sgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnter9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape^gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
[gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Rgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopStackPop[gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*2
_output_shapes 
:���������:���������
�
:gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectSelect=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:���������
�
<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
7gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumSum:gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectIgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
	keep_dims( *

Tidx0
�
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeReshape7gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������*
Tshape0
�
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1Sum<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Kgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
	keep_dims( *

Tidx0
�
=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape_1Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
_output_shapes
: *
Tshape0
�
4gradients/lstm_1/while/Switch_3_grad_1/NextIterationNextIteration+gradients/lstm_1/while/mul_4_grad/Reshape_1*'
_class
loc:@lstm_1/while/Merge_3*
T0*'
_output_shapes
:���������
�
7gradients/lstm_1/while/clip_by_value/Minimum_grad/ShapeShapelstm_1/while/add_1*
out_type0*
T0*
_output_shapes
:*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
�
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB *
_output_shapes
: 
�
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Shape1gradients/lstm_1/while/clip_by_value_grad/Reshape*
out_type0*
T0*
_output_shapes
:*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum
�
=gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *    *
_output_shapes
: 
�
7gradients/lstm_1/while/clip_by_value/Minimum_grad/zerosFill9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2=gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/Const*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
Agradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_accStack*

stack_name *N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnterRefEnterAgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Egradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush	StackPushDgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_1^gradients/Add*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Mgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterAgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Dgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopStackPopMgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*'
_output_shapes
:���������
�
Cgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1RefEnterCgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush_1	StackPushFgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_1^gradients/Add*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Ogradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1StackPopOgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
: 
�
;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual	LessEqualDgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopFgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
Mgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
Pgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterMgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Qgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnter7gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape^gradients/Add*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Ygradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterMgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Pgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopStackPopYgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*2
_output_shapes 
:���������:���������
�
8gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectSelect;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot
LogicalNot;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:���������
�
:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Select<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
5gradients/lstm_1/while/clip_by_value/Minimum_grad/SumSum8gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectGgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs*
_output_shapes
:*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
	keep_dims( *

Tidx0
�
9gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeReshape5gradients/lstm_1/while/clip_by_value/Minimum_grad/SumPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������*
Tshape0
�
7gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_1Sum:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Igradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*
_output_shapes
:*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
	keep_dims( *

Tidx0
�
;gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape_1Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_19gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
_output_shapes
: *
Tshape0
�
'gradients/lstm_1/while/add_4_grad/ShapeShapelstm_1/while/strided_slice_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_4
�
)gradients/lstm_1/while/add_4_grad/Shape_1Shapelstm_1/while/MatMul_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_4
�
=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
�
Agradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_4_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_4*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
�
@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
�
Cgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_4_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_4*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
�
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_4*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_4_grad/SumSum)gradients/lstm_1/while/Tanh_grad/TanhGrad7gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_4*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/add_4_grad/ReshapeReshape%gradients/lstm_1/while/add_4_grad/Sum@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_4*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/add_4_grad/Sum_1Sum)gradients/lstm_1/while/Tanh_grad/TanhGrad9gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_4*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/add_4_grad/Reshape_1Reshape'gradients/lstm_1/while/add_4_grad/Sum_1Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_4*
T0*'
_output_shapes
:���������*
Tshape0
�
1gradients/lstm_1/while/strided_slice_3_grad/ShapeShapelstm_1/while/TensorArrayReadV3*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/strided_slice_3
�
Bgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_3*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_3_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
is_constant(
�
Egradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_3/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_3/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_3/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
	elem_type0*
_output_shapes
:
�
<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_6_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_3
�
1gradients/lstm_1/while/MatMul_3_grad/MatMul/EnterEnterlstm_1/strided_slice_7*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_3*
is_constant(
�
+gradients/lstm_1/while/MatMul_3_grad/MatMulMatMul+gradients/lstm_1/while/add_6_grad/Reshape_11gradients/lstm_1/while/MatMul_3_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_3*
T0*'
_output_shapes
:���������
�
3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
�
6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
is_constant(
�
7gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnterlstm_1/while/mul_7^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
�
?gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
is_constant(
�
6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
	elem_type0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_6_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_3*
T0*
_output_shapes

:
�
'gradients/lstm_1/while/add_3_grad/ShapeShapelstm_1/while/mul_3*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_3
�
)gradients/lstm_1/while/add_3_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_3*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_3*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_3*
is_constant(
�
Agradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_3_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_3*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_3*
is_constant(
�
@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_3*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_3_grad/Shape_1*%
_class
loc:@lstm_1/while/add_3*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_3_grad/SumSum;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape7gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_3*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/add_3_grad/ReshapeReshape%gradients/lstm_1/while/add_3_grad/Sum@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_3*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/add_3_grad/Sum_1Sum;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape9gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_3*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/add_3_grad/Reshape_1Reshape'gradients/lstm_1/while/add_3_grad/Sum_1)gradients/lstm_1/while/add_3_grad/Shape_1*%
_class
loc:@lstm_1/while/add_3*
T0*
_output_shapes
: *
Tshape0
�
'gradients/lstm_1/while/add_1_grad/ShapeShapelstm_1/while/mul_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_1
�
)gradients/lstm_1/while/add_1_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_1*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_1*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_1*
is_constant(
�
Agradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_1_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_1*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_1*
is_constant(
�
@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_1_grad/Shape_1*%
_class
loc:@lstm_1/while/add_1*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_1_grad/SumSum9gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape7gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_1*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/add_1_grad/ReshapeReshape%gradients/lstm_1/while/add_1_grad/Sum@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_1*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/add_1_grad/Sum_1Sum9gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape9gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_1*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/add_1_grad/Reshape_1Reshape'gradients/lstm_1/while/add_1_grad/Sum_1)gradients/lstm_1/while/add_1_grad/Shape_1*%
_class
loc:@lstm_1/while/add_1*
T0*
_output_shapes
: *
Tshape0
�
1gradients/lstm_1/while/strided_slice_2_grad/ShapeShapelstm_1/while/TensorArrayReadV3*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/strided_slice_2
�
Bgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_2*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_2_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
is_constant(
�
Egradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_2/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_2/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_2/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
	elem_type0*
_output_shapes
:
�
<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_4_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_2
�
1gradients/lstm_1/while/MatMul_2_grad/MatMul/EnterEnterlstm_1/strided_slice_6*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_2*
is_constant(
�
+gradients/lstm_1/while/MatMul_2_grad/MatMulMatMul+gradients/lstm_1/while/add_4_grad/Reshape_11gradients/lstm_1/while/MatMul_2_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_2*
T0*'
_output_shapes
:���������
�
3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
�
6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
is_constant(
�
7gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnterlstm_1/while/mul_5^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
T0*
swap_memory(*
_output_shapes
:
�
?gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
is_constant(
�
6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
	elem_type0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/MatMul_2_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_4_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_2*
T0*
_output_shapes

:
�
'gradients/lstm_1/while/mul_7_grad/ShapeShapelstm_1/while/Identity_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_7
�
)gradients/lstm_1/while/mul_7_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_7*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_7*
is_constant(
�
Agradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_7_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_7*
is_constant(
�
@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_7_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_7*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_7_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_7_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_7_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
is_constant(
�
/gradients/lstm_1/while/mul_7_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_7_grad/mul/RefEnterlstm_1/while/mul_7/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_7_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_7_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
is_constant(
�
.gradients/lstm_1/while/mul_7_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_7_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
	elem_type0*
_output_shapes
: 
�
%gradients/lstm_1/while/mul_7_grad/mulMul+gradients/lstm_1/while/MatMul_3_grad/MatMul.gradients/lstm_1/while/mul_7_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_7_grad/SumSum%gradients/lstm_1/while/mul_7_grad/mul7gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_7*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_7_grad/ReshapeReshape%gradients/lstm_1/while/mul_7_grad/Sum@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������*
Tshape0
�
-gradients/lstm_1/while/mul_7_grad/mul_1/f_accStack*

stack_name *C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_7_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_7_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
is_constant(
�
1gradients/lstm_1/while/mul_7_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_7_grad/mul_1/RefEnterlstm_1/while/Identity_2^gradients/Add*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_7_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_7_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
is_constant(
�
0gradients/lstm_1/while/mul_7_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_7_grad/mul_1/StackPop/RefEnter^gradients/Sub*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
	elem_type0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_7_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_3_grad/MatMul*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_7_grad/Sum_1Sum'gradients/lstm_1/while/mul_7_grad/mul_19gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_7*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_7_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_7_grad/Sum_1)gradients/lstm_1/while/mul_7_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_7*
T0*
_output_shapes
: *
Tshape0
�
0gradients/lstm_1/while/MatMul_3/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
valueB*    *
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
is_constant( 
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_3/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0* 
_output_shapes
:: *
N
�
1gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*(
_output_shapes
::
�
.gradients/lstm_1/while/MatMul_3/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_3_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
�
8gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_3/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
�
'gradients/lstm_1/while/mul_3_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_3*
valueB *
_output_shapes
: 
�
)gradients/lstm_1/while/mul_3_grad/Shape_1Shapelstm_1/while/add_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_3
�
=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_3*
is_constant(
�
Agradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_3_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_3*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_3*
is_constant(
�
@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_3_grad/Shape@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_3_grad/mul/f_accStack*

stack_name *>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_3_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_3_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
is_constant(
�
/gradients/lstm_1/while/mul_3_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_3_grad/mul/RefEnterlstm_1/while/add_2^gradients/Add*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_3_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_3_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
is_constant(
�
.gradients/lstm_1/while/mul_3_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_3_grad/mul/StackPop/RefEnter^gradients/Sub*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_3_grad/mulMul)gradients/lstm_1/while/add_3_grad/Reshape.gradients/lstm_1/while/mul_3_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_3_grad/SumSum%gradients/lstm_1/while/mul_3_grad/mul7gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_3*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_3_grad/ReshapeReshape%gradients/lstm_1/while/mul_3_grad/Sum'gradients/lstm_1/while/mul_3_grad/Shape*%
_class
loc:@lstm_1/while/mul_3*
T0*
_output_shapes
: *
Tshape0
�
-gradients/lstm_1/while/mul_3_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_3_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_3_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
is_constant(
�
1gradients/lstm_1/while/mul_3_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_3_grad/mul_1/RefEnterlstm_1/while/mul_3/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_3_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_3_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
is_constant(
�
0gradients/lstm_1/while/mul_3_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_3_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
	elem_type0*
_output_shapes
: 
�
'gradients/lstm_1/while/mul_3_grad/mul_1Mul0gradients/lstm_1/while/mul_3_grad/mul_1/StackPop)gradients/lstm_1/while/add_3_grad/Reshape*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_3_grad/Sum_1Sum'gradients/lstm_1/while/mul_3_grad/mul_19gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_3*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_3_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_3_grad/Sum_1@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_1*
valueB *
_output_shapes
: 
�
)gradients/lstm_1/while/mul_1_grad/Shape_1Shapelstm_1/while/add*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_1
�
=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_1*
is_constant(
�
Agradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_1_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_1*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_1*
is_constant(
�
@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_1_grad/Shape@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_1_grad/mul/f_accStack*

stack_name *<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_1_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_1_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
is_constant(
�
/gradients/lstm_1/while/mul_1_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_1_grad/mul/RefEnterlstm_1/while/add^gradients/Add*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_1_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_1_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
is_constant(
�
.gradients/lstm_1/while/mul_1_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_1_grad/mulMul)gradients/lstm_1/while/add_1_grad/Reshape.gradients/lstm_1/while/mul_1_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_1_grad/SumSum%gradients/lstm_1/while/mul_1_grad/mul7gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_1*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_1_grad/ReshapeReshape%gradients/lstm_1/while/mul_1_grad/Sum'gradients/lstm_1/while/mul_1_grad/Shape*%
_class
loc:@lstm_1/while/mul_1*
T0*
_output_shapes
: *
Tshape0
�
-gradients/lstm_1/while/mul_1_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_1_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_1_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
is_constant(
�
1gradients/lstm_1/while/mul_1_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_1_grad/mul_1/RefEnterlstm_1/while/mul_1/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_1_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_1_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
is_constant(
�
0gradients/lstm_1/while/mul_1_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
	elem_type0*
_output_shapes
: 
�
'gradients/lstm_1/while/mul_1_grad/mul_1Mul0gradients/lstm_1/while/mul_1_grad/mul_1/StackPop)gradients/lstm_1/while/add_1_grad/Reshape*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_1_grad/Sum_1Sum'gradients/lstm_1/while/mul_1_grad/mul_19gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_1*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_1_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_1_grad/Sum_1@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/mul_5_grad/ShapeShapelstm_1/while/Identity_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_5
�
)gradients/lstm_1/while/mul_5_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_5*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_5*
is_constant(
�
Agradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_5_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_5*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_5*
is_constant(
�
@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_5_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_5*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_5_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_5_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_5_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
is_constant(
�
/gradients/lstm_1/while/mul_5_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_5_grad/mul/RefEnterlstm_1/while/mul_5/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_5_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_5_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
is_constant(
�
.gradients/lstm_1/while/mul_5_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_5_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
	elem_type0*
_output_shapes
: 
�
%gradients/lstm_1/while/mul_5_grad/mulMul+gradients/lstm_1/while/MatMul_2_grad/MatMul.gradients/lstm_1/while/mul_5_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_5_grad/SumSum%gradients/lstm_1/while/mul_5_grad/mul7gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_5*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_5_grad/ReshapeReshape%gradients/lstm_1/while/mul_5_grad/Sum@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/mul_5_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_2_grad/MatMul*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_5_grad/Sum_1Sum'gradients/lstm_1/while/mul_5_grad/mul_19gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_5*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_5_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_5_grad/Sum_1)gradients/lstm_1/while/mul_5_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_5*
T0*
_output_shapes
: *
Tshape0
�
0gradients/lstm_1/while/MatMul_2/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
valueB*    *
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
is_constant( 
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_2/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0* 
_output_shapes
:: *
N
�
1gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*(
_output_shapes
::
�
.gradients/lstm_1/while/MatMul_2/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_2_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
�
8gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_2/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
�
+gradients/lstm_1/strided_slice_7_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_7*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_22gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_7
�
'gradients/lstm_1/while/add_2_grad/ShapeShapelstm_1/while/strided_slice_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_2
�
)gradients/lstm_1/while/add_2_grad/Shape_1Shapelstm_1/while/MatMul_1*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_2
�
=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
�
Agradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_2_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_2*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
�
@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
�
Cgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_2_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_2*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
�
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_2*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_2_grad/SumSum+gradients/lstm_1/while/mul_3_grad/Reshape_17gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_2*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/add_2_grad/ReshapeReshape%gradients/lstm_1/while/add_2_grad/Sum@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/add_2*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/add_2_grad/Sum_1Sum+gradients/lstm_1/while/mul_3_grad/Reshape_19gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/add_2*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/add_2_grad/Reshape_1Reshape'gradients/lstm_1/while/add_2_grad/Sum_1Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_2*
T0*'
_output_shapes
:���������*
Tshape0
�
%gradients/lstm_1/while/add_grad/ShapeShapelstm_1/while/strided_slice*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/while/add
�
'gradients/lstm_1/while/add_grad/Shape_1Shapelstm_1/while/MatMul*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/while/add
�
;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_accStack*

stack_name *#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
�
>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnterRefEnter;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
�
?gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush	StackPush>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter%gradients/lstm_1/while/add_grad/Shape^gradients/Add*#
_class
loc:@lstm_1/while/add*
T0*
swap_memory(*
_output_shapes
:
�
Ggradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
�
>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPopStackPopGgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
�
=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1RefEnter=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
�
Agradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush_1	StackPush@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1'gradients/lstm_1/while/add_grad/Shape_1^gradients/Add*#
_class
loc:@lstm_1/while/add*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
�
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1StackPopIgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
�
5gradients/lstm_1/while/add_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1*#
_class
loc:@lstm_1/while/add*
T0*2
_output_shapes 
:���������:���������
�
#gradients/lstm_1/while/add_grad/SumSum+gradients/lstm_1/while/mul_1_grad/Reshape_15gradients/lstm_1/while/add_grad/BroadcastGradientArgs*
_output_shapes
:*#
_class
loc:@lstm_1/while/add*
T0*
	keep_dims( *

Tidx0
�
'gradients/lstm_1/while/add_grad/ReshapeReshape#gradients/lstm_1/while/add_grad/Sum>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop*#
_class
loc:@lstm_1/while/add*
T0*'
_output_shapes
:���������*
Tshape0
�
%gradients/lstm_1/while/add_grad/Sum_1Sum+gradients/lstm_1/while/mul_1_grad/Reshape_17gradients/lstm_1/while/add_grad/BroadcastGradientArgs:1*
_output_shapes
:*#
_class
loc:@lstm_1/while/add*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/add_grad/Reshape_1Reshape%gradients/lstm_1/while/add_grad/Sum_1@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1*#
_class
loc:@lstm_1/while/add*
T0*'
_output_shapes
:���������*
Tshape0
�
+gradients/lstm_1/strided_slice_6_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_6*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_22gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_6
�
1gradients/lstm_1/while/strided_slice_1_grad/ShapeShapelstm_1/while/TensorArrayReadV3*
out_type0*
T0*
_output_shapes
:*/
_class%
#!loc:@lstm_1/while/strided_slice_1
�
Bgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_1*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_1_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
is_constant(
�
Egradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_1/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_1/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_1/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
	elem_type0*
_output_shapes
:
�
<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_2_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_1
�
1gradients/lstm_1/while/MatMul_1_grad/MatMul/EnterEnterlstm_1/strided_slice_5*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_1*
is_constant(
�
+gradients/lstm_1/while/MatMul_1_grad/MatMulMatMul+gradients/lstm_1/while/add_2_grad/Reshape_11gradients/lstm_1/while/MatMul_1_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_1*
T0*'
_output_shapes
:���������
�
3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
�
6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
is_constant(
�
7gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnterlstm_1/while/mul_2^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
T0*
swap_memory(*
_output_shapes
:
�
?gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
is_constant(
�
6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
	elem_type0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_2_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_1*
T0*
_output_shapes

:
�
/gradients/lstm_1/while/strided_slice_grad/ShapeShapelstm_1/while/TensorArrayReadV3*
out_type0*
T0*
_output_shapes
:*-
_class#
!loc:@lstm_1/while/strided_slice
�
@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_accStack*

stack_name *-
_class#
!loc:@lstm_1/while/strided_slice*
	elem_type0*
_output_shapes
:
�
Cgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnterRefEnter@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/strided_slice*
is_constant(
�
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush	StackPushCgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter/gradients/lstm_1/while/strided_slice_grad/Shape^gradients/Add*-
_class#
!loc:@lstm_1/while/strided_slice*
T0*
swap_memory(*
_output_shapes
:
�
Lgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop/RefEnterRefEnter@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/strided_slice*
is_constant(
�
Cgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopStackPopLgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*-
_class#
!loc:@lstm_1/while/strided_slice*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_1	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1 lstm_1/while/strided_slice/stack^gradients/Add*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
is_constant(
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_2	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2"lstm_1/while/strided_slice/stack_1^gradients/Add*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
is_constant(
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_3	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3"lstm_1/while/strided_slice/stack_2^gradients/Add*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
is_constant(
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
	elem_type0*
_output_shapes
:
�
:gradients/lstm_1/while/strided_slice_grad/StridedSliceGradStridedSliceGradCgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3'gradients/lstm_1/while/add_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *-
_class#
!loc:@lstm_1/while/strided_slice
�
/gradients/lstm_1/while/MatMul_grad/MatMul/EnterEnterlstm_1/strided_slice_4*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/MatMul*
is_constant(
�
)gradients/lstm_1/while/MatMul_grad/MatMulMatMul)gradients/lstm_1/while/add_grad/Reshape_1/gradients/lstm_1/while/MatMul_grad/MatMul/Enter*
transpose_b(*
transpose_a( *&
_class
loc:@lstm_1/while/MatMul*
T0*'
_output_shapes
:���������
�
1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_accStack*

stack_name *=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
�
4gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnterRefEnter1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
is_constant(
�
5gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPush	StackPush4gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnterlstm_1/while/mul^gradients/Add*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
T0*
swap_memory(*
_output_shapes
:
�
=gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnter1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
is_constant(
�
4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPopStackPop=gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
	elem_type0*'
_output_shapes
:���������
�
+gradients/lstm_1/while/MatMul_grad/MatMul_1MatMul4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop)gradients/lstm_1/while/add_grad/Reshape_1*
transpose_b( *
transpose_a(*&
_class
loc:@lstm_1/while/MatMul*
T0*
_output_shapes

:
�
'gradients/lstm_1/while/mul_2_grad/ShapeShapelstm_1/while/Identity_2*
out_type0*
T0*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_2
�
)gradients/lstm_1/while/mul_2_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_2*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_2*
is_constant(
�
Agradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_2_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_2*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_2*
is_constant(
�
@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_2_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_2*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_2_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_2_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_2_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
is_constant(
�
/gradients/lstm_1/while/mul_2_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_2_grad/mul/RefEnterlstm_1/while/mul_2/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_2_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_2_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
is_constant(
�
.gradients/lstm_1/while/mul_2_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
	elem_type0*
_output_shapes
: 
�
%gradients/lstm_1/while/mul_2_grad/mulMul+gradients/lstm_1/while/MatMul_1_grad/MatMul.gradients/lstm_1/while/mul_2_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_2_grad/SumSum%gradients/lstm_1/while/mul_2_grad/mul7gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_2*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_2_grad/ReshapeReshape%gradients/lstm_1/while/mul_2_grad/Sum@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:���������*
Tshape0
�
'gradients/lstm_1/while/mul_2_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_1_grad/MatMul*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_2_grad/Sum_1Sum'gradients/lstm_1/while/mul_2_grad/mul_19gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*%
_class
loc:@lstm_1/while/mul_2*
T0*
	keep_dims( *

Tidx0
�
+gradients/lstm_1/while/mul_2_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_2_grad/Sum_1)gradients/lstm_1/while/mul_2_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_2*
T0*
_output_shapes
: *
Tshape0
�
0gradients/lstm_1/while/MatMul_1/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
valueB*    *
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
is_constant( 
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_1/Enter_grad/NextIteration*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0* 
_output_shapes
:: *
N
�
1gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*(
_output_shapes
::
�
.gradients/lstm_1/while/MatMul_1/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_1_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
�
8gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_1/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
�
gradients/AddN_3AddN<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad:gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*'
_output_shapes
:���������`*
N
�
Ugradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
�
Wgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
�
Ogradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Ugradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterWgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
source	gradients*'
_class
loc:@lstm_1/TensorArray_1*
_output_shapes

::
�
Kgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityWgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1P^gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
Qgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ogradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopgradients/AddN_3Kgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
%gradients/lstm_1/while/mul_grad/ShapeShapelstm_1/while/Identity_2*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/while/mul
�
'gradients/lstm_1/while/mul_grad/Shape_1Const^gradients/Sub*
dtype0*#
_class
loc:@lstm_1/while/mul*
valueB *
_output_shapes
: 
�
;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_accStack*

stack_name *#
_class
loc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
�
>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnterRefEnter;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/mul*
is_constant(
�
?gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPush	StackPush>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnter%gradients/lstm_1/while/mul_grad/Shape^gradients/Add*#
_class
loc:@lstm_1/while/mul*
T0*
swap_memory(*
_output_shapes
:
�
Ggradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/mul*
is_constant(
�
>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopStackPopGgradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
�
5gradients/lstm_1/while/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop'gradients/lstm_1/while/mul_grad/Shape_1*#
_class
loc:@lstm_1/while/mul*
T0*2
_output_shapes 
:���������:���������
�
)gradients/lstm_1/while/mul_grad/mul/f_accStack*

stack_name *<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
	elem_type0*
_output_shapes
:
�
,gradients/lstm_1/while/mul_grad/mul/RefEnterRefEnter)gradients/lstm_1/while/mul_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
is_constant(
�
-gradients/lstm_1/while/mul_grad/mul/StackPush	StackPush,gradients/lstm_1/while/mul_grad/mul/RefEnterlstm_1/while/mul/y^gradients/Add*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
T0*
swap_memory(*
_output_shapes
:
�
5gradients/lstm_1/while/mul_grad/mul/StackPop/RefEnterRefEnter)gradients/lstm_1/while/mul_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
is_constant(
�
,gradients/lstm_1/while/mul_grad/mul/StackPopStackPop5gradients/lstm_1/while/mul_grad/mul/StackPop/RefEnter^gradients/Sub*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
	elem_type0*
_output_shapes
: 
�
#gradients/lstm_1/while/mul_grad/mulMul)gradients/lstm_1/while/MatMul_grad/MatMul,gradients/lstm_1/while/mul_grad/mul/StackPop*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:���������
�
#gradients/lstm_1/while/mul_grad/SumSum#gradients/lstm_1/while/mul_grad/mul5gradients/lstm_1/while/mul_grad/BroadcastGradientArgs*
_output_shapes
:*#
_class
loc:@lstm_1/while/mul*
T0*
	keep_dims( *

Tidx0
�
'gradients/lstm_1/while/mul_grad/ReshapeReshape#gradients/lstm_1/while/mul_grad/Sum>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:���������*
Tshape0
�
%gradients/lstm_1/while/mul_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop)gradients/lstm_1/while/MatMul_grad/MatMul*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_grad/Sum_1Sum%gradients/lstm_1/while/mul_grad/mul_17gradients/lstm_1/while/mul_grad/BroadcastGradientArgs:1*
_output_shapes
:*#
_class
loc:@lstm_1/while/mul*
T0*
	keep_dims( *

Tidx0
�
)gradients/lstm_1/while/mul_grad/Reshape_1Reshape%gradients/lstm_1/while/mul_grad/Sum_1'gradients/lstm_1/while/mul_grad/Shape_1*#
_class
loc:@lstm_1/while/mul*
T0*
_output_shapes
: *
Tshape0
�
.gradients/lstm_1/while/MatMul/Enter_grad/b_accConst*
dtype0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
valueB*    *
_output_shapes

:
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter.gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
is_constant( 
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2Merge0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_16gradients/lstm_1/while/MatMul/Enter_grad/NextIteration*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0* 
_output_shapes
:: *
N
�
/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*(
_output_shapes
::
�
,gradients/lstm_1/while/MatMul/Enter_grad/AddAdd1gradients/lstm_1/while/MatMul/Enter_grad/Switch:1+gradients/lstm_1/while/MatMul_grad/MatMul_1*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
�
6gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration,gradients/lstm_1/while/MatMul/Enter_grad/Add*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit/gradients/lstm_1/while/MatMul/Enter_grad/Switch*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
�
+gradients/lstm_1/strided_slice_5_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_5*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_22gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_5
�
;gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*
valueB
 *    *
_output_shapes
: 
�
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter;gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant( 
�
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Cgradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIteration*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: : *
N
�
<gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: : 
�
9gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/AddAdd>gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch:1Qgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
Cgradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration9gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Add*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit<gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
gradients/AddN_4AddN)gradients/lstm_1/while/mul_7_grad/Reshape)gradients/lstm_1/while/mul_5_grad/Reshape)gradients/lstm_1/while/mul_2_grad/Reshape'gradients/lstm_1/while/mul_grad/Reshape*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������*
N
�
+gradients/lstm_1/strided_slice_4_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_4*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_20gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_4
�
rgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArray_1=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*'
_class
loc:@lstm_1/TensorArray_1*
_output_shapes

::
�
ngradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3s^gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
dgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3rgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lstm_1/TensorArrayUnstack/rangengradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*+
_output_shapes
:0���������`*
element_shape:
�
4gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_4*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:���������
�
gradients/AddN_5AddN6gradients/lstm_1/strided_slice_7_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_6_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_5_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*)
_class
loc:@lstm_1/strided_slice_7*
T0*
_output_shapes

:`*
N
�
1gradients/lstm_1/transpose_grad/InvertPermutationInvertPermutationlstm_1/transpose/perm*#
_class
loc:@lstm_1/transpose*
T0*
_output_shapes
:
�
)gradients/lstm_1/transpose_grad/transpose	Transposedgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV31gradients/lstm_1/transpose_grad/InvertPermutation*
Tperm0*
T0*+
_output_shapes
:���������0`*#
_class
loc:@lstm_1/transpose
�
!gradients/lstm_1/concat_grad/RankConst*
dtype0* 
_class
loc:@lstm_1/concat*
value	B :*
_output_shapes
: 
�
 gradients/lstm_1/concat_grad/modFloorModlstm_1/concat/axis!gradients/lstm_1/concat_grad/Rank* 
_class
loc:@lstm_1/concat*
T0*
_output_shapes
: 
�
"gradients/lstm_1/concat_grad/ShapeShapelstm_1/Reshape_1*
out_type0*
T0*
_output_shapes
:* 
_class
loc:@lstm_1/concat
�
#gradients/lstm_1/concat_grad/ShapeNShapeNlstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7*
out_type0* 
_class
loc:@lstm_1/concat*
T0*,
_output_shapes
::::*
N
�
)gradients/lstm_1/concat_grad/ConcatOffsetConcatOffset gradients/lstm_1/concat_grad/mod#gradients/lstm_1/concat_grad/ShapeN%gradients/lstm_1/concat_grad/ShapeN:1%gradients/lstm_1/concat_grad/ShapeN:2%gradients/lstm_1/concat_grad/ShapeN:3* 
_class
loc:@lstm_1/concat*,
_output_shapes
::::*
N
�
"gradients/lstm_1/concat_grad/SliceSlice)gradients/lstm_1/transpose_grad/transpose)gradients/lstm_1/concat_grad/ConcatOffset#gradients/lstm_1/concat_grad/ShapeN*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������0
�
$gradients/lstm_1/concat_grad/Slice_1Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:1%gradients/lstm_1/concat_grad/ShapeN:1*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������0
�
$gradients/lstm_1/concat_grad/Slice_2Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:2%gradients/lstm_1/concat_grad/ShapeN:2*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������0
�
$gradients/lstm_1/concat_grad/Slice_3Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:3%gradients/lstm_1/concat_grad/ShapeN:3*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������0
�
%gradients/lstm_1/Reshape_1_grad/ShapeShapelstm_1/BiasAdd*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/Reshape_1
�
'gradients/lstm_1/Reshape_1_grad/ReshapeReshape"gradients/lstm_1/concat_grad/Slice%gradients/lstm_1/Reshape_1_grad/Shape*#
_class
loc:@lstm_1/Reshape_1*
T0*'
_output_shapes
:���������*
Tshape0
�
%gradients/lstm_1/Reshape_3_grad/ShapeShapelstm_1/BiasAdd_1*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/Reshape_3
�
'gradients/lstm_1/Reshape_3_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_1%gradients/lstm_1/Reshape_3_grad/Shape*#
_class
loc:@lstm_1/Reshape_3*
T0*'
_output_shapes
:���������*
Tshape0
�
%gradients/lstm_1/Reshape_5_grad/ShapeShapelstm_1/BiasAdd_2*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/Reshape_5
�
'gradients/lstm_1/Reshape_5_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_2%gradients/lstm_1/Reshape_5_grad/Shape*#
_class
loc:@lstm_1/Reshape_5*
T0*'
_output_shapes
:���������*
Tshape0
�
%gradients/lstm_1/Reshape_7_grad/ShapeShapelstm_1/BiasAdd_3*
out_type0*
T0*
_output_shapes
:*#
_class
loc:@lstm_1/Reshape_7
�
'gradients/lstm_1/Reshape_7_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_3%gradients/lstm_1/Reshape_7_grad/Shape*#
_class
loc:@lstm_1/Reshape_7*
T0*'
_output_shapes
:���������*
Tshape0
�
)gradients/lstm_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_1_grad/Reshape*!
_class
loc:@lstm_1/BiasAdd*
T0*
_output_shapes
:*
data_formatNHWC
�
+gradients/lstm_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_3_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_1*
T0*
_output_shapes
:*
data_formatNHWC
�
+gradients/lstm_1/BiasAdd_2_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_5_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_2*
T0*
_output_shapes
:*
data_formatNHWC
�
+gradients/lstm_1/BiasAdd_3_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_7_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_3*
T0*
_output_shapes
:*
data_formatNHWC
�
#gradients/lstm_1/MatMul_grad/MatMulMatMul'gradients/lstm_1/Reshape_1_grad/Reshapelstm_1/strided_slice*
transpose_b(*
transpose_a( * 
_class
loc:@lstm_1/MatMul*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/MatMul_grad/MatMul_1MatMullstm_1/Reshape'gradients/lstm_1/Reshape_1_grad/Reshape*
transpose_b( *
transpose_a(* 
_class
loc:@lstm_1/MatMul*
T0*
_output_shapes

:
�
+gradients/lstm_1/strided_slice_8_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_8*
valueB:`*
_output_shapes
:
�
6gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2)gradients/lstm_1/BiasAdd_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:`*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_8
�
%gradients/lstm_1/MatMul_1_grad/MatMulMatMul'gradients/lstm_1/Reshape_3_grad/Reshapelstm_1/strided_slice_1*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_1*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/MatMul_1_grad/MatMul_1MatMullstm_1/Reshape_2'gradients/lstm_1/Reshape_3_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_1*
T0*
_output_shapes

:
�
+gradients/lstm_1/strided_slice_9_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_9*
valueB:`*
_output_shapes
:
�
6gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2+gradients/lstm_1/BiasAdd_1_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:`*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_9
�
%gradients/lstm_1/MatMul_2_grad/MatMulMatMul'gradients/lstm_1/Reshape_5_grad/Reshapelstm_1/strided_slice_2*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_2*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/MatMul_2_grad/MatMul_1MatMullstm_1/Reshape_4'gradients/lstm_1/Reshape_5_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_2*
T0*
_output_shapes

:
�
,gradients/lstm_1/strided_slice_10_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_10*
valueB:`*
_output_shapes
:
�
7gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2+gradients/lstm_1/BiasAdd_2_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:`*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask **
_class 
loc:@lstm_1/strided_slice_10
�
%gradients/lstm_1/MatMul_3_grad/MatMulMatMul'gradients/lstm_1/Reshape_7_grad/Reshapelstm_1/strided_slice_3*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_3*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/MatMul_3_grad/MatMul_1MatMullstm_1/Reshape_6'gradients/lstm_1/Reshape_7_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_3*
T0*
_output_shapes

:
�
,gradients/lstm_1/strided_slice_11_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_11*
valueB:`*
_output_shapes
:
�
7gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2+gradients/lstm_1/BiasAdd_3_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:`*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask **
_class 
loc:@lstm_1/strided_slice_11
�
)gradients/lstm_1/strided_slice_grad/ShapeConst*
dtype0*'
_class
loc:@lstm_1/strided_slice*
valueB"   `   *
_output_shapes
:
�
4gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad)gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2%gradients/lstm_1/MatMul_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *'
_class
loc:@lstm_1/strided_slice
�
+gradients/lstm_1/strided_slice_1_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_1*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2'gradients/lstm_1/MatMul_1_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_1
�
+gradients/lstm_1/strided_slice_2_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_2*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2'gradients/lstm_1/MatMul_2_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_2
�
+gradients/lstm_1/strided_slice_3_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_3*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2'gradients/lstm_1/MatMul_3_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_3
�
gradients/AddN_6AddN6gradients/lstm_1/strided_slice_8_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_9_grad/StridedSliceGrad7gradients/lstm_1/strided_slice_10_grad/StridedSliceGrad7gradients/lstm_1/strided_slice_11_grad/StridedSliceGrad*)
_class
loc:@lstm_1/strided_slice_8*
T0*
_output_shapes
:`*
N
�
gradients/AddN_7AddN"gradients/lstm_1/Square_grad/mul_14gradients/lstm_1/strided_slice_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_1_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_2_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_3_grad/StridedSliceGrad* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:`*
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
�
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
valueB*    *
_output_shapes

:
~

Variable_1
VariableV2*
dtype0*
shape
:*
	container *
shared_name *
_output_shapes

:
�
Variable_1/AssignAssign
Variable_1Const_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes

:
T
Const_4Const*
dtype0*
valueB`*    *
_output_shapes
:`
v

Variable_2
VariableV2*
dtype0*
shape:`*
	container *
shared_name *
_output_shapes
:`
�
Variable_2/AssignAssign
Variable_2Const_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:`
k
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes
:`
\
Const_5Const*
dtype0*
valueB`*    *
_output_shapes

:`
~

Variable_3
VariableV2*
dtype0*
shape
:`*
	container *
shared_name *
_output_shapes

:`
�
Variable_3/AssignAssign
Variable_3Const_5*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes

:`
o
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes

:`
\
Const_6Const*
dtype0*
valueB`*    *
_output_shapes

:`
~

Variable_4
VariableV2*
dtype0*
shape
:`*
	container *
shared_name *
_output_shapes

:`
�
Variable_4/AssignAssign
Variable_4Const_6*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:`
o
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes

:`
J
mul_2Mulrho/readVariable/read*
T0*
_output_shapes
:
L
sub_1/xConst*
dtype0*
valueB
 *  �?*
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
�
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
 *  �*
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
 *w�+2*
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
�
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

:
L
sub_3/xConst*
dtype0*
valueB
 *  �?*
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

:
F
mul_6Mulsub_3Square_2*
T0*
_output_shapes

:
C
add_4Addmul_5mul_6*
T0*
_output_shapes

:
�
Assign_2Assign
Variable_1add_4*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:
f
mul_7Mullr/read&gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
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
 *  �*
_output_shapes
: 
\
clip_by_value_1/MinimumMinimumadd_4Const_10*
T0*
_output_shapes

:
e
clip_by_value_1Maximumclip_by_value_1/MinimumConst_9*
T0*
_output_shapes

:
H
Sqrt_1Sqrtclip_by_value_1*
T0*
_output_shapes

:
L
add_5/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
F
add_5AddSqrt_1add_5/y*
T0*
_output_shapes

:
G
div_2RealDivmul_7add_5*
T0*
_output_shapes

:
Q
sub_4Subdense_1/kernel/readdiv_2*
T0*
_output_shapes

:
�
Assign_3Assigndense_1/kernelsub_4*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:
L
mul_8Mulrho/readVariable_2/read*
T0*
_output_shapes
:`
L
sub_5/xConst*
dtype0*
valueB
 *  �?*
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
:`
B
mul_9Mulsub_5Square_3*
T0*
_output_shapes
:`
?
add_6Addmul_8mul_9*
T0*
_output_shapes
:`
�
Assign_4Assign
Variable_2add_6*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:`
M
mul_10Mullr/readgradients/AddN_6*
T0*
_output_shapes
:`
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
 *  �*
_output_shapes
: 
X
clip_by_value_2/MinimumMinimumadd_6Const_12*
T0*
_output_shapes
:`
b
clip_by_value_2Maximumclip_by_value_2/MinimumConst_11*
T0*
_output_shapes
:`
D
Sqrt_2Sqrtclip_by_value_2*
T0*
_output_shapes
:`
L
add_7/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
B
add_7AddSqrt_2add_7/y*
T0*
_output_shapes
:`
D
div_3RealDivmul_10add_7*
T0*
_output_shapes
:`
J
sub_6Sublstm_1/bias/readdiv_3*
T0*
_output_shapes
:`
�
Assign_5Assignlstm_1/biassub_6*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes
:`
Q
mul_11Mulrho/readVariable_3/read*
T0*
_output_shapes

:`
L
sub_7/xConst*
dtype0*
valueB
 *  �?*
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

:`
G
mul_12Mulsub_7Square_4*
T0*
_output_shapes

:`
E
add_8Addmul_11mul_12*
T0*
_output_shapes

:`
�
Assign_6Assign
Variable_3add_8*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes

:`
Q
mul_13Mullr/readgradients/AddN_7*
T0*
_output_shapes

:`
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
 *  �*
_output_shapes
: 
\
clip_by_value_3/MinimumMinimumadd_8Const_14*
T0*
_output_shapes

:`
f
clip_by_value_3Maximumclip_by_value_3/MinimumConst_13*
T0*
_output_shapes

:`
H
Sqrt_3Sqrtclip_by_value_3*
T0*
_output_shapes

:`
L
add_9/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
F
add_9AddSqrt_3add_9/y*
T0*
_output_shapes

:`
H
div_4RealDivmul_13add_9*
T0*
_output_shapes

:`
P
sub_8Sublstm_1/kernel/readdiv_4*
T0*
_output_shapes

:`
�
Assign_7Assignlstm_1/kernelsub_8*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes

:`
Q
mul_14Mulrho/readVariable_4/read*
T0*
_output_shapes

:`
L
sub_9/xConst*
dtype0*
valueB
 *  �?*
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

:`
G
mul_15Mulsub_9Square_5*
T0*
_output_shapes

:`
F
add_10Addmul_14mul_15*
T0*
_output_shapes

:`
�
Assign_8Assign
Variable_4add_10*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:`
Q
mul_16Mullr/readgradients/AddN_5*
T0*
_output_shapes

:`
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
 *  �*
_output_shapes
: 
]
clip_by_value_4/MinimumMinimumadd_10Const_16*
T0*
_output_shapes

:`
f
clip_by_value_4Maximumclip_by_value_4/MinimumConst_15*
T0*
_output_shapes

:`
H
Sqrt_4Sqrtclip_by_value_4*
T0*
_output_shapes

:`
M
add_11/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
H
add_11AddSqrt_4add_11/y*
T0*
_output_shapes

:`
I
div_5RealDivmul_16add_11*
T0*
_output_shapes

:`
[
sub_10Sublstm_1/recurrent_kernel/readdiv_5*
T0*
_output_shapes

:`
�
Assign_9Assignlstm_1/recurrent_kernelsub_10*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes

:`
�
group_deps_1NoOp^add_1^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
�
initNoOp^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^lstm_1/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign
^lr/Assign^rho/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign"F��uc�     ;"�	'09G�AJ��
�-�-
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
2	��
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
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
2	�
�
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
2	�
<
Mul
x"T
y"T
z"T"
Ttype:
2	�
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
�
A
Placeholder
output"dtype"
dtypetype"
shapeshape: 
�
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
2	�
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
�
RefEnter
data"T�
output"T�"	
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
handle�"
	elem_typetype"

stack_namestring �
?
StackPop
handle�
elem"	elem_type"
	elem_typetype
V
	StackPush
handle�	
elem"T
output"T"	
Ttype"
swap_memorybool( 
�
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
�
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
�
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
sourcestring�
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
�
TensorArrayV3
size

handle
flow"
dtypetype"
element_shapeshape:"
dynamic_sizebool( "
clear_after_readbool("
tensor_array_namestring �
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
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype*1.0.12v1.0.0-65-g4763edf-dirty�
b
lstm_1_inputPlaceholder*
dtype0*
shape: *+
_output_shapes
:���������0
l
lstm_1/random_uniform/shapeConst*
dtype0*
valueB"   `   *
_output_shapes
:
^
lstm_1/random_uniform/minConst*
dtype0*
valueB
 *O�~�*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
dtype0*
valueB
 *O�~>*
_output_shapes
: 
�
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
dtype0*
seed2���*
seed���)*
T0*
_output_shapes

:`
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
T0*
_output_shapes
: 
�
lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
T0*
_output_shapes

:`
{
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes

:`
�
lstm_1/kernel
VariableV2*
dtype0*
shape
:`*
shared_name *
	container *
_output_shapes

:`
�
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes

:`
x
lstm_1/kernel/readIdentitylstm_1/kernel* 
_class
loc:@lstm_1/kernel*
T0*
_output_shapes

:`
T
lstm_1/SquareSquarelstm_1/kernel/read*
T0*
_output_shapes

:`
Q
lstm_1/mul/xConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
W

lstm_1/mulMullstm_1/mul/xlstm_1/Square*
T0*
_output_shapes

:`
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
�H
%lstm_1/recurrent_kernel/initial_valueConst*
dtype0*�H
value�HB�H`"�H��F��]�=���9��=,�U����C�>(��}�P���&>5G,��R�=蓳=M�P���V�]脽
g�;Ѝ	=�xb������%�0½�����0�[1���Ƚɋ����C>. �=�B<S�=���*=�8j�O����μ�u�K�=��f<RH�,�n�x�.>B�6=�K�<��h�=�P>B'���>���E>c�!�������>��=:����=!LO�?��ƧڼO	�==�=��#�!uP�Z ��J�4>3q@=	��<�]ʼe��=+�<<�����S����/:0��y�� �?��[+�	n��RP������y�?w">%�V�=�1��g9���F=����V�:n�ͼUO��:U�:
*���G>�1�$�޼��Ľe�v���=qq�=µ�����<9��:m��|#�+�?W�z�l��н*꼅~H>z�$�Fጾ�8M<a���gO,>���=ʽ}�=@؎��c�=�^>��v{=�YR=�B�;��<��=��=Lba�d�>�=�0��� ��-ռ�o�=Y|����<�R>iٽ�r>єC��刽��Z�m�p=�Ƚ�=�3�<K&�G�r��	U�B��=z{\�;¼��;P��"�f=
>���|��A�%>=��N�=�`�=�㱽$6=<w>�c��z�h��=H�Z���S��.I��r�=`J��J�̽=��=�N�`��;K�*�46����;㬬�*T���p�J.��>��=����A/+��E�=xa=��=I3�=w+�=�.|�6x>qjE=G�<eE@�>o�== ����J�s>�>	�㽟W=�1�=����(y�0�o=~n�<���<*�t�2Є=�p:�,�<ԥp=�=��>�6=���v�z����н/"��+6!�ʁ.������|��6����;������p�ս�=?>i��=�g�e�=��=�,�ː����=? =}K�=�a���΅<�� ��LӼ9��=q�=�}<�&��&>��3:`*S��Ǚ��O=ƿ3�ň�=Q:�>�+�=� ֽBT{��!G�`����=&��f��F~=ݼ���󥽡�(��ڻ�V����A #>��۽Se�c1=�M<�G����<��&���=�Zu�/�=wü�F���K[ۼh&�=,ޡ=]E��ɳ�IC\���>��%>Uzj<��H���<_�лܻ�FW#=9*�=d:��i7�����������=J� =�r�=&/K��N<=o���ד<��>�n=?�Y�v��=��$>����`�=U�ƽ�î��>j�༢~�=m>�Y=v�=�q�=���:��=�<��_����=�8��K���.N<^WǼ�`�<l�=M�=����!=㶡=F��˓ >�d��'9�񘉾xؼ-`!>xI����U=�5�Ŋ>NĐ=e����Hq��-�3D= Ԡ=��e=�[ǽ����n�=�$>w�=0���ɲ!�]��8t�<��=��=!7�B8��S�����=3�d�_�f<]��"[2>gM#�^h�;����@s�s�?�0��=��>	��<�j�=|����>�9=���?�B��v<҄Ƽ+a=�%�}M�<�@3���=�b��A=��
�rz=)n)=Y��C>c�r���K>���d�ȕ��ۓ=3�<�j9�~�=�~�=����zhj�>G =e���g�=,��>m�=/�=Y�=��_��N=��;ϐ�=A�><[�=/�h���)="a��t�M>���= ��<�ծ=��l���<Y.�u;��]>MF;�$(�=k����=�G"�
׌>��\=+T�=��8����;I���j,=����5�x�Q��d�<QO�<�&=�b��m��Di����=��Z<Z�U���Q?�=X���мIB�`j)=N�=���Is���>�w&��U�=z��<Gnm�y�S,�:Y��=2��?H�=}8a�7P�>��f�a���r�9>I����=e}d= ��<<Y"�rY}=6<�ߑ�[��<��;���w,;��½����;��By��cf�<��;&޽��ֽUE�=�K>��=�f%��	>�q�ĝ��Җ>9	�=y�:>`e�;S����=U�H�'�<q��i�=u�����>#��>����R�=f�>���=��=q�?>�7;=��;���<�
�=��<���=��<��=��&�WW}�b��=��Q=�\���Ds���߽j >�����O��'�a=-�g<8b >N��-8��B����fǉ>}:L��Q��@	>��=�	>��.>�)>���6B���r=�����B�<b��=��]�._ƽ=�a�1G;wb�L��Ͻ��<��->8H�=W���L��ٚ=�ћ��(>��9��ǵ;^��"3��M�ݽ�����R#�$X>�;�=�C����W�B�(ڜ�m��=�SS���==t5��F>ͼF��;���<����������j��Y����=�k\>���~�˾�[���e<Uj<�T��s����$���ܽ�/����=��v=�4�=t$>�����1�l�н���<�-�<p�=�b=��V<�?���KF>��L��5>�4X����sӦ=8ڣ=L%�=�緽1�=Щ�
�#=<��;o�=Cv6>���x��d�<h>�O�?H={L���=5A�:�b>�	>��>��>����Pz�= E���=��Q1�+�4<�J���>��D=c�E���"�6_���p�=X�=7Y<�OG��>���[g��;�A�=�0潟����h=u�k=ψ,�]/>i>�㌼�9n�z�<q ��ܨ;{"�gF������=����>�_[=͍�RR�����kWʼ�aܽ��>�=�n�=�p:����I����
P=z�k�l+���y����=vk��������=�[�=;�ӽj���i�=���b:��W�+�u&1��ؽ��;��>���=�-9���<�佝��<�>-Wo=բI=��l=��r�7 �=�`������h=����Uy=�V�K]=ֳ��s'�c�>wK=�V�.3>x�]�!��K�=n�X�$�8<r���"ܽ��Y=��o�B�?>հV�H�3=�`�=K'�=˛q=��M��s>Cm�=��G�Kb]=��&>m`d��g����B�PP���`M>�e��8=P�� ˼���;��<�-�;U���g=�^�<���;!<�*�<���=c}��W��<,��<L�3�t=�Kt=2����<T&��lƨ=���<�d��P���-���>%�Gd��j��H���ǅ��6=Ӵ���K��*��=����T��=�[-�Fm=�d�=U`D=M�(�����f�~�ؽ��=�@
�#+�A��1>�f>7z���:���:01���_���>���;��̼h�ܽwB_�/"�����(G���c��ޞ�;��:=�n���=!�=o߼�2 *=N�=
�V���ݽ�#�=ӳs�� �=&?ؼ�=Y�ܽ��6'F=Q�����Z�<`�<=<DA�6ݯ=B>=�Q#>t�Y>�[��f��=��>G>Ƚ��ӽ.Ky�]ia������d�����+�<�k���<�k�->!�������A�Ƌ{�q����fJ=��m��j���c*���������8M>�>�1�uн~�=vet��o�=��=�;=��<���<Yg�=C��<i�=�r���9`>��U<H�/<��=��:��!o=�>#���r��B`&<	*��f�ѳ�=c=2�>�Մ<jN۽ �8��>~�=?�O��%���զ�|,�=y���\Ȑ�";"x�mQ������R+ڼ~0����>pѽX�W���x�!J�<��D��_=n�-8�l��㮽����>�;=�,?�hJO�S'C��4�l����ʽ
�*>�$$=z�T��U���l=�`���X=���=c˹=<��=*2;���;H�<���=�Jh�A��=dL/��T =���m���� Y��8����j�"X��/��d�|>&�=��;>O�۽Hэ�ݜ�*(����@/f<�糽i�=���<�d�=wG�=A���q��7>��'>���=���9���=UB��r�=G�)>f#�[��<��=����F^��y�=��콎��<��g�=����󽽓�=P�F>��=K��o'<�����`�=�倽r����I��^����=�C�=�ㅻ��k<��=>�����P=1+s>��p�`v�=\_=Z�">��~�=/>;°<T���ۻ�@�x=tz=kB=v�w���=�L5�S�c=?�v=�A�f�@�AHv>:�S<�=��w����<eBn=��e=�vo��B�<�5�<���}�<C�<�5�=�rF=��Ǽl�=/��;��'>�煽a�<=�&�=�1>�w~=�<:=No�;�r���r$=�~��Cܽ�̽eb0>ߦ<4��鱫�mf��0S���Խ����L���<=<�k�Hq =3������<�=�<sj�=�9���b�?�n�i��;F��=��U�b�F�m]=jl��򣘽�,�=��K���'��u=�!4>�{����q=B�U=hr�<?Ah���y�DV=�ʸ=6��΍�=G`s>�>�ω�,�@�=l�����BO>�ѻ�f=w>;|�-=h��.�<�9�=mLt��d����;<&��=U���L����>���<<}�G��
`�=�]���D=���=y�"�`���2�w9���C>s!�=�ə=�;�<g���o=B8 ��� >��,��ӑ��:�=L�����L>���<ȧ�=˾[�ʦ��|�!�G>RH\=��<=�9Z<���<�|P���'p�<K.P�����z���q��=|�D=V��ڪ>8�
�(�"��n���o0>['ѽ>�Y�^j�=Tֵ=�Q���=۽X�X���>%����@�=�E���*ʽ-�;��q�=��ݽř��<�=#'�=mʸ��G���0��[���'/��|�=1����)�;�O�<�g=(­�-�[���>�r�|;������짻�� w<?x�=��`=Ȃ~=��2����"��+���?>Ο�=�����3�=�����x�=Wv�=0U�+�$=�t���Y�=3a�=p�z=x�H=��>�U=����	\=|�w�c�D>�,�=��=V�#�v�%����D�=>�a� ΋�� �=t<��;��C=7k��_f�=�1)>���=u5��Y� PF�CMỻf��������I�v=u��=4[>��<o� �	0=ς���f<�kN�X!7�*�=T ���m���<B6�=2�D���>𮩽�>n��=�0�(�����y���<Z=��m�����p>��=����T>�]R���l=�l��iB�=/B�=*(�=�>�	�=�>��K�Eｧ])=-���}{���ͽޗC�K��ғ�=c`-��j<h����e=Z�4=�)>�Ҝ<�m,��D��,�:\���=-|=LTO��a��W=�ٚ=���=|M>����K���m�9�����2>a����]=񉒽���v[�=R-���/�ي2=��=��=��� >���Hĸ=�����=My��x�M�ҽF}߽��<�U�=6�.=�������=��1>��!�x��=(<~�խ
>`G/>�5t<焀��,�&^[=�3�<c��|�=V�=c������Og >��	=@k���L!��_(���>'mN��W>_3�����)/�_�J!�=�d����h��=��d>���=@h�=��U>3�4=�[=���=K>:*�O����M�=(�f�L������G6=�Ƽ�^>k)޼���=.t���>�ý�;�=�����>��<�~1=�3�c~Q���ǽ�6�=� �ҙ�=6��!]�=+:�=�R=^X��]��wɑ�.��<�,>��6���Z�D|��@�=|Y<��>�י>��1b<�?�<\����vE��d>�b�=�˽V�U=4F�v�J���<�; �+��=sj=j!�9�5�=LC���="4>���=�q����9>���:o���o]^����=�C�w����=�-�Z�>Q$�=�VۼTd���O>V�W	\��&
�ɓ#�%Yv=k����L��/V�������������m�=
՗<���`��<i��<��2=6�=&>����I��x�<��<=�F���X<�S����<�= > 8齶I����èh=`���U>W��L��<-�R��$�=�M=�E��_�l�=]�=:5꽓�>Y���du�}�M�ur>�����E�ӂ� �=kgC�G� >x
�=>��Li3<Ϲ0����=ts>ļH=��=��O<�<2�߼i��!��=9tj>������=�W�=�"��K>��<� �M[S�B
 >x�����9>��:��P�=|]V>��
�\*�=�:.�?og�1~->K��BI�;�B�0�=4ـ���	>wZ8��㼞��=m�>��F>2����8��V~:; ����x�b�7�;2mr=?M�<�}:��һy�>�r�<U�p=aMV<��=���ߤ��}������0�<�"���qG=:�A>-��<� �cbd�*�>W����j�Z�6�P��r�]=��=t����0�ۣ?�Y*��o�2��V'��#*>���� �^<U&���= 駼F��=�1#�����3��� <��<����0��]Y�=(�� J�=8O>~Z$�yJ�Nlb� ·�|-��"�=��;����9�=�ڍ=��s��H>���=��s����׾=ϧd�1����=�b>hm����'>���:��	�6�M<�`��oQ>�>�V�=f��;����=H#>!��=h�V�oZ��	�=^�P�j�r�N��K�@�^;���,e�=�&>{C�;iQs�0u�=�+/�Uܲ���2���9��=:Uн^31=�hN� ������$��6�=�=r��.\(=?;�9��=��ý�Ҽ������۽�3O<5�S=۟|�H~:�;;��S�<�����!�sN8<���^�=KZ>��	��8��g.=��Nl��*<=��=� �=�Aa<���b[:=d�����=Uj�=�.�<�+=yڔ=S�bU��re9&!ҽ l��(2=n:H=.5�<(D@�H�=�%<hBǽ'D�;�=�� ��=(��aػ@(�>Vx>�޽k�=�T>w{E=|���J#��ؗ=/1�=���<Dռ D>��&�<��)�:>�^�=3��=�f���G��h��49E>7h�;]�;�<��,���4�	�
=u0
��P�=��<�\T��F>�}�=���=��'=�#�=�/�=)%�>W�}<�	>��޽�"k<n#>�	���=ɨ�;���=�O=ڈs�
�v=��=)�&=)Z#�[}j�H�ս�>���c��=�fi=�[��3�'��	�=��=<Y��5�p=�Z�#�;�J�2=��ĽL��0��=����4��=��H>	u.>Ag=uY�=����,�^=%�r��a=%����ݽ4$>�<�=�G=(=#R2���>���@�>�q����<xN�;(��w�=��� ̇=a��<MWa�ך�=�w��_%=>~��J��vX�=��Ž�=�cͼ��<��=X">�=U�N>�x���U�;Pý��=l��aQy�ma�<���`^k>���=�J!>S���ςc�"�y.��rÉ���r>گ�<��2����D��=$��@6���6=���=g��;`r����	<�,��Z�ս긽4�ý#��=��G��!>�a>�$>�� �<�:�[JѼ�(�=�[=q�ļ��T=^��=T�1�J�2>j�Y�+i��ռ'��<m�o��H�<��=���=Z�"/K��΁=�%�=e�����=�KT�z��=��<��
<��i�BKd��$^=��<�?�<�C�$V�]ҽ#�ͽAO�=P�=��<מ�=��=�!�9�=�����+>7A�<���<�(l�xSI�2q��}q>� Լ�`m=:𫽄�=�=zbK<U�sV�=�擽}���#���-�$M=z�0=-1�����$���Z"<�6>�����j��IL<"+(>�ý�0';+\�=6�,��V�P��=6�ٻ42]=�~�=mE���d%=����>����B=h�ͽu���m=�ko�;;r#>`G=�k>�ռ�܌��1�<D=�Gy=\�=�[̼��<�|��.�O>z	�;㹹<�p=�z8�Uh�=h�$=�gн(�޺��:��^.=�*��>]�b��gr��o2<̽M�5,�<�N���<�f��<������������<v��=0Mݽ�F�=���=�|�=J�*������=�U@��޼0�=c1�<ue=�Zh�J�4��<�<��[>����=��=���=&�7>X�=>.l'>0���;��#�ٽ�P=���=|�ƽw,���*��A&>�-G=S�qOH��xd=6)>���<A����6�=��==�A�y��<��E��醽���=��,�(L�q)�:��� +���3���>e�@<��=5�O=0���)=̬b<�]u���B=Q³�Q�P�a�=�0���=�q�=y0�Q��f��T�>ｼ=�H��G�6>�G�2�ﶋ=]<>���=���=�Žoo�=u[��OA��>��Լ6$�=!J�=�D�<��I�K�*>���<���=��%> �>�A;>��ؽ��㽴<��S&�mWA�
4�~^�WV��7��x=�<=s��;�� =��:X=�;���y*=>-���N&�����=@�j���v!
�:'���� �=��:�p8q�j<�=��j����=+���-5��_=������=��E�A�8<�&n�2�3�Ƿx��x�޽�(D�is�]�=���l���������o��V��$Xǽ.[`=�/�=F��譢=E�<U;�=����}�=��4�����I��<���Fs����=2v����?>�$#�g��=���=_ֽs^=�k�S]=&=W�I�V>l��=w����(�iڽh2�<�'>d�=.��=�,���>�&>k��=,݄=��ٻ��Q�ف�< m=%t��r#r��x>�ϻ���$�ǹ;<��=��q����Kܚ=^d�={@��
��Ρ��Ƅ���*
_output_shapes

:`
�
lstm_1/recurrent_kernel
VariableV2*
dtype0*
shape
:`*
shared_name *
	container *
_output_shapes

:`
�
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes

:`
�
lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
T0*
_output_shapes

:`
[
lstm_1/Const_1Const*
dtype0*
valueB`*    *
_output_shapes
:`
w
lstm_1/bias
VariableV2*
dtype0*
shape:`*
shared_name *
	container *
_output_shapes
:`
�
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/Const_1*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes
:`
n
lstm_1/bias/readIdentitylstm_1/bias*
_class
loc:@lstm_1/bias*
T0*
_output_shapes
:`
k
lstm_1/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
m
lstm_1/strided_slice/stack_1Const*
dtype0*
valueB"       *
_output_shapes
:
m
lstm_1/strided_slice/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_sliceStridedSlicelstm_1/kernel/readlstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_1/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_1Const*
dtype0*
valueB"    0   *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_1StridedSlicelstm_1/kernel/readlstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_2/stackConst*
dtype0*
valueB"    0   *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
dtype0*
valueB"    H   *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_2StridedSlicelstm_1/kernel/readlstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_3/stackConst*
dtype0*
valueB"    H   *
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
�
lstm_1/strided_slice_3StridedSlicelstm_1/kernel/readlstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

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
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_4/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_4StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_5/stackConst*
dtype0*
valueB"       *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_1Const*
dtype0*
valueB"    0   *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_5StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_6/stackConst*
dtype0*
valueB"    0   *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
dtype0*
valueB"    H   *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_6StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_7/stackConst*
dtype0*
valueB"    H   *
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
�
lstm_1/strided_slice_7StridedSlicelstm_1/recurrent_kernel/readlstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_2*
new_axis_mask *
Index0*
_output_shapes

:*

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
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_8/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
lstm_1/strided_slice_8StridedSlicelstm_1/bias/readlstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
f
lstm_1/strided_slice_9/stackConst*
dtype0*
valueB:*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
dtype0*
valueB:0*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
lstm_1/strided_slice_9StridedSlicelstm_1/bias/readlstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_10/stackConst*
dtype0*
valueB:0*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
dtype0*
valueB:H*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
�
lstm_1/strided_slice_10StridedSlicelstm_1/bias/readlstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_11/stackConst*
dtype0*
valueB:H*
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
�
lstm_1/strided_slice_11StridedSlicelstm_1/bias/readlstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2*
new_axis_mask *
Index0*
_output_shapes
:*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
b
lstm_1/zeros_like	ZerosLikelstm_1_input*
T0*+
_output_shapes
:���������0
o
lstm_1/Sum_1/reduction_indicesConst*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/Sum_1Sumlstm_1/zeros_likelstm_1/Sum_1/reduction_indices*

Tidx0*
T0*
	keep_dims( *#
_output_shapes
:���������
`
lstm_1/ExpandDims/dimConst*
dtype0*
valueB :
���������*
_output_shapes
: 
�
lstm_1/ExpandDims
ExpandDimslstm_1/Sum_1lstm_1/ExpandDims/dim*

Tdim0*
T0*'
_output_shapes
:���������
f
lstm_1/Tile/multiplesConst*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:���������
e
lstm_1/Reshape/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
}
lstm_1/ReshapeReshapelstm_1_inputlstm_1/Reshape/shape*
Tshape0*
T0*'
_output_shapes
:���������
�
lstm_1/MatMulMatMullstm_1/Reshapelstm_1/strided_slice*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*
data_formatNHWC*
T0*'
_output_shapes
:���������
a
lstm_1/stackConst*
dtype0*!
valueB"����0      *
_output_shapes
:
}
lstm_1/Reshape_1Reshapelstm_1/BiasAddlstm_1/stack*
Tshape0*
T0*+
_output_shapes
:���������0
g
lstm_1/Reshape_2/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
lstm_1/Reshape_2Reshapelstm_1_inputlstm_1/Reshape_2/shape*
Tshape0*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_1MatMullstm_1/Reshape_2lstm_1/strided_slice_1*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*
data_formatNHWC*
T0*'
_output_shapes
:���������
c
lstm_1/stack_1Const*
dtype0*!
valueB"����0      *
_output_shapes
:
�
lstm_1/Reshape_3Reshapelstm_1/BiasAdd_1lstm_1/stack_1*
Tshape0*
T0*+
_output_shapes
:���������0
g
lstm_1/Reshape_4/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
lstm_1/Reshape_4Reshapelstm_1_inputlstm_1/Reshape_4/shape*
Tshape0*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_2MatMullstm_1/Reshape_4lstm_1/strided_slice_2*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*
data_formatNHWC*
T0*'
_output_shapes
:���������
c
lstm_1/stack_2Const*
dtype0*!
valueB"����0      *
_output_shapes
:
�
lstm_1/Reshape_5Reshapelstm_1/BiasAdd_2lstm_1/stack_2*
Tshape0*
T0*+
_output_shapes
:���������0
g
lstm_1/Reshape_6/shapeConst*
dtype0*
valueB"����   *
_output_shapes
:
�
lstm_1/Reshape_6Reshapelstm_1_inputlstm_1/Reshape_6/shape*
Tshape0*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_3MatMullstm_1/Reshape_6lstm_1/strided_slice_3*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*
data_formatNHWC*
T0*'
_output_shapes
:���������
c
lstm_1/stack_3Const*
dtype0*!
valueB"����0      *
_output_shapes
:
�
lstm_1/Reshape_7Reshapelstm_1/BiasAdd_3lstm_1/stack_3*
Tshape0*
T0*+
_output_shapes
:���������0
T
lstm_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
lstm_1/concatConcatV2lstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7lstm_1/concat/axis*+
_output_shapes
:���������0`*

Tidx0*
T0*
N
j
lstm_1/transpose/permConst*
dtype0*!
valueB"          *
_output_shapes
:
�
lstm_1/transpose	Transposelstm_1/concatlstm_1/transpose/perm*
Tperm0*
T0*+
_output_shapes
:0���������`
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
�
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
�
lstm_1/strided_slice_13StridedSlicelstm_1/transposelstm_1/strided_slice_13/stacklstm_1/strided_slice_13/stack_1lstm_1/strided_slice_13/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

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
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_14/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_14StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_14/stacklstm_1/strided_slice_14/stack_1lstm_1/strided_slice_14/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_15/stackConst*
dtype0*
valueB"       *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_1Const*
dtype0*
valueB"    0   *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_15StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_15/stacklstm_1/strided_slice_15/stack_1lstm_1/strided_slice_15/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_16/stackConst*
dtype0*
valueB"    0   *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_1Const*
dtype0*
valueB"    H   *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_2Const*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/strided_slice_16StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_16/stacklstm_1/strided_slice_16/stack_1lstm_1/strided_slice_16/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_17/stackConst*
dtype0*
valueB"    H   *
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
�
lstm_1/strided_slice_17StridedSlicelstm_1/strided_slice_13lstm_1/strided_slice_17/stacklstm_1/strided_slice_17/stack_1lstm_1/strided_slice_17/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
S
lstm_1/mul_1/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
b
lstm_1/mul_1Mullstm_1/Tilelstm_1/mul_1/y*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_4MatMullstm_1/mul_1lstm_1/strided_slice_4*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
lstm_1/add_1Addlstm_1/strided_slice_14lstm_1/MatMul_4*
T0*'
_output_shapes
:���������
S
lstm_1/mul_2/xConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
c
lstm_1/mul_2Mullstm_1/mul_2/xlstm_1/add_1*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
w
lstm_1/clip_by_value/MinimumMinimumlstm_1/add_2lstm_1/Const_3*
T0*'
_output_shapes
:���������

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_2*
T0*'
_output_shapes
:���������
S
lstm_1/mul_3/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
b
lstm_1/mul_3Mullstm_1/Tilelstm_1/mul_3/y*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_5MatMullstm_1/mul_3lstm_1/strided_slice_5*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
lstm_1/add_3Addlstm_1/strided_slice_15lstm_1/MatMul_5*
T0*'
_output_shapes
:���������
S
lstm_1/mul_4/xConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
c
lstm_1/mul_4Mullstm_1/mul_4/xlstm_1/add_3*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
y
lstm_1/clip_by_value_1/MinimumMinimumlstm_1/add_4lstm_1/Const_5*
T0*'
_output_shapes
:���������
�
lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_4*
T0*'
_output_shapes
:���������
j
lstm_1/mul_5Mullstm_1/clip_by_value_1lstm_1/Tile*
T0*'
_output_shapes
:���������
S
lstm_1/mul_6/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
b
lstm_1/mul_6Mullstm_1/Tilelstm_1/mul_6/y*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_6MatMullstm_1/mul_6lstm_1/strided_slice_6*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
lstm_1/add_5Addlstm_1/strided_slice_16lstm_1/MatMul_6*
T0*'
_output_shapes
:���������
S
lstm_1/TanhTanhlstm_1/add_5*
T0*'
_output_shapes
:���������
h
lstm_1/mul_7Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:���������
a
lstm_1/add_6Addlstm_1/mul_5lstm_1/mul_7*
T0*'
_output_shapes
:���������
S
lstm_1/mul_8/yConst*
dtype0*
valueB
 *  �?*
_output_shapes
: 
b
lstm_1/mul_8Mullstm_1/Tilelstm_1/mul_8/y*
T0*'
_output_shapes
:���������
�
lstm_1/MatMul_7MatMullstm_1/mul_8lstm_1/strided_slice_7*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
o
lstm_1/add_7Addlstm_1/strided_slice_17lstm_1/MatMul_7*
T0*'
_output_shapes
:���������
S
lstm_1/mul_9/xConst*
dtype0*
valueB
 *��L>*
_output_shapes
: 
c
lstm_1/mul_9Mullstm_1/mul_9/xlstm_1/add_7*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
y
lstm_1/clip_by_value_2/MinimumMinimumlstm_1/add_8lstm_1/Const_7*
T0*'
_output_shapes
:���������
�
lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_6*
T0*'
_output_shapes
:���������
U
lstm_1/Tanh_1Tanhlstm_1/add_6*
T0*'
_output_shapes
:���������
m
lstm_1/mul_10Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:���������
�
lstm_1/TensorArrayTensorArrayV3lstm_1/strided_slice_12*
_output_shapes

::*
dtype0*
dynamic_size( *
clear_after_read(* 
tensor_array_name	output_ta*
element_shape:
�
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
�
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
�
lstm_1/TensorArrayUnstack/rangeRange%lstm_1/TensorArrayUnstack/range/start'lstm_1/TensorArrayUnstack/strided_slice%lstm_1/TensorArrayUnstack/range/delta*

Tidx0*#
_output_shapes
:���������
�
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
�
lstm_1/while/EnterEnterlstm_1/time*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes
: 
�
lstm_1/while/Enter_1Enterlstm_1/TensorArray:1*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes
:
�
lstm_1/while/Enter_2Enterlstm_1/Tile*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*'
_output_shapes
:���������
�
lstm_1/while/Enter_3Enterlstm_1/Tile*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*'
_output_shapes
:���������
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
�
lstm_1/while/Merge_2Mergelstm_1/while/Enter_2lstm_1/while/NextIteration_2*
N*
T0*)
_output_shapes
:���������: 
�
lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*
N*
T0*)
_output_shapes
:���������: 
�
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
�
lstm_1/while/SwitchSwitchlstm_1/while/Mergelstm_1/while/LoopCond*%
_class
loc:@lstm_1/while/Merge*
T0*
_output_shapes
: : 
�
lstm_1/while/Switch_1Switchlstm_1/while/Merge_1lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes

::
�
lstm_1/while/Switch_2Switchlstm_1/while/Merge_2lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_2*
T0*:
_output_shapes(
&:���������:���������
�
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:���������:���������
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
:���������
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*
T0*'
_output_shapes
:���������
�
$lstm_1/while/TensorArrayReadV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
�
&lstm_1/while/TensorArrayReadV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: **

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
�
lstm_1/while/TensorArrayReadV3TensorArrayReadV3$lstm_1/while/TensorArrayReadV3/Enterlstm_1/while/Identity&lstm_1/while/TensorArrayReadV3/Enter_1*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*'
_output_shapes
:���������`
�
 lstm_1/while/strided_slice/stackConst^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:
�
"lstm_1/while/strided_slice/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:
�
"lstm_1/while/strided_slice/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/while/strided_sliceStridedSlicelstm_1/while/TensorArrayReadV3 lstm_1/while/strided_slice/stack"lstm_1/while/strided_slice/stack_1"lstm_1/while/strided_slice/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_1/stackConst^lstm_1/while/Identity*
dtype0*
valueB"       *
_output_shapes
:
�
$lstm_1/while/strided_slice_1/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    0   *
_output_shapes
:
�
$lstm_1/while/strided_slice_1/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/while/strided_slice_1StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_1/stack$lstm_1/while/strided_slice_1/stack_1$lstm_1/while/strided_slice_1/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_2/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    0   *
_output_shapes
:
�
$lstm_1/while/strided_slice_2/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    H   *
_output_shapes
:
�
$lstm_1/while/strided_slice_2/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/while/strided_slice_2StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_2/stack$lstm_1/while/strided_slice_2/stack_1$lstm_1/while/strided_slice_2/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_3/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    H   *
_output_shapes
:
�
$lstm_1/while/strided_slice_3/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:
�
$lstm_1/while/strided_slice_3/stack_2Const^lstm_1/while/Identity*
dtype0*
valueB"      *
_output_shapes
:
�
lstm_1/while/strided_slice_3StridedSlicelstm_1/while/TensorArrayReadV3"lstm_1/while/strided_slice_3/stack$lstm_1/while/strided_slice_3/stack_1$lstm_1/while/strided_slice_3/stack_2*
new_axis_mask *
Index0*'
_output_shapes
:���������*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
o
lstm_1/while/mul/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  �?*
_output_shapes
: 
v
lstm_1/while/mulMullstm_1/while/Identity_2lstm_1/while/mul/y*
T0*'
_output_shapes
:���������
�
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice_4*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:
�
lstm_1/while/MatMulMatMullstm_1/while/mullstm_1/while/MatMul/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
z
lstm_1/while/addAddlstm_1/while/strided_slicelstm_1/while/MatMul*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_1/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *��L>*
_output_shapes
: 
s
lstm_1/while/mul_1Mullstm_1/while/mul_1/xlstm_1/while/add*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
�
"lstm_1/while/clip_by_value/MinimumMinimumlstm_1/while/add_1lstm_1/while/Const_1*
T0*'
_output_shapes
:���������
�
lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_2/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  �?*
_output_shapes
: 
z
lstm_1/while/mul_2Mullstm_1/while/Identity_2lstm_1/while/mul_2/y*
T0*'
_output_shapes
:���������
�
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_5*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:
�
lstm_1/while/MatMul_1MatMullstm_1/while/mul_2lstm_1/while/MatMul_1/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/while/add_2Addlstm_1/while/strided_slice_1lstm_1/while/MatMul_1*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_3/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *��L>*
_output_shapes
: 
u
lstm_1/while/mul_3Mullstm_1/while/mul_3/xlstm_1/while/add_2*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
�
$lstm_1/while/clip_by_value_1/MinimumMinimumlstm_1/while/add_3lstm_1/while/Const_3*
T0*'
_output_shapes
:���������
�
lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*
T0*'
_output_shapes
:���������
�
lstm_1/while/mul_4Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_3*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_5/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  �?*
_output_shapes
: 
z
lstm_1/while/mul_5Mullstm_1/while/Identity_2lstm_1/while/mul_5/y*
T0*'
_output_shapes
:���������
�
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_6*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:
�
lstm_1/while/MatMul_2MatMullstm_1/while/mul_5lstm_1/while/MatMul_2/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/while/add_4Addlstm_1/while/strided_slice_2lstm_1/while/MatMul_2*
T0*'
_output_shapes
:���������
_
lstm_1/while/TanhTanhlstm_1/while/add_4*
T0*'
_output_shapes
:���������
z
lstm_1/while/mul_6Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*
T0*'
_output_shapes
:���������
s
lstm_1/while/add_5Addlstm_1/while/mul_4lstm_1/while/mul_6*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_7/yConst^lstm_1/while/Identity*
dtype0*
valueB
 *  �?*
_output_shapes
: 
z
lstm_1/while/mul_7Mullstm_1/while/Identity_2lstm_1/while/mul_7/y*
T0*'
_output_shapes
:���������
�
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_7*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:
�
lstm_1/while/MatMul_3MatMullstm_1/while/mul_7lstm_1/while/MatMul_3/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
lstm_1/while/add_6Addlstm_1/while/strided_slice_3lstm_1/while/MatMul_3*
T0*'
_output_shapes
:���������
q
lstm_1/while/mul_8/xConst^lstm_1/while/Identity*
dtype0*
valueB
 *��L>*
_output_shapes
: 
u
lstm_1/while/mul_8Mullstm_1/while/mul_8/xlstm_1/while/add_6*
T0*'
_output_shapes
:���������
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
:���������
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
 *  �?*
_output_shapes
: 
�
$lstm_1/while/clip_by_value_2/MinimumMinimumlstm_1/while/add_7lstm_1/while/Const_5*
T0*'
_output_shapes
:���������
�
lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:���������
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_5*
T0*'
_output_shapes
:���������
~
lstm_1/while/mul_9Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:���������
�
6lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/TensorArray*
is_constant(
�
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
�
lstm_1/while/NextIteration_1NextIteration0lstm_1/while/TensorArrayWrite/TensorArrayWriteV3*
T0*
_output_shapes
: 
s
lstm_1/while/NextIteration_2NextIterationlstm_1/while/mul_9*
T0*'
_output_shapes
:���������
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/add_5*
T0*'
_output_shapes
:���������
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
:���������
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*
T0*'
_output_shapes
:���������
�
)lstm_1/TensorArrayStack/TensorArraySizeV3TensorArraySizeV3lstm_1/TensorArraylstm_1/while/Exit_1*%
_class
loc:@lstm_1/TensorArray*
_output_shapes
: 
�
#lstm_1/TensorArrayStack/range/startConst*
dtype0*%
_class
loc:@lstm_1/TensorArray*
value	B : *
_output_shapes
: 
�
#lstm_1/TensorArrayStack/range/deltaConst*
dtype0*%
_class
loc:@lstm_1/TensorArray*
value	B :*
_output_shapes
: 
�
lstm_1/TensorArrayStack/rangeRange#lstm_1/TensorArrayStack/range/start)lstm_1/TensorArrayStack/TensorArraySizeV3#lstm_1/TensorArrayStack/range/delta*%
_class
loc:@lstm_1/TensorArray*

Tidx0*#
_output_shapes
:���������
�
+lstm_1/TensorArrayStack/TensorArrayGatherV3TensorArrayGatherV3lstm_1/TensorArraylstm_1/TensorArrayStack/rangelstm_1/while/Exit_1*$
element_shape:���������*
dtype0*%
_class
loc:@lstm_1/TensorArray*4
_output_shapes"
 :������������������
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
�
lstm_1/TensorArrayReadV3TensorArrayReadV3lstm_1/TensorArray
lstm_1/sublstm_1/while/Exit_1*
dtype0*%
_class
loc:@lstm_1/TensorArray*'
_output_shapes
:���������
l
lstm_1/transpose_1/permConst*
dtype0*!
valueB"          *
_output_shapes
:
�
lstm_1/transpose_1	Transpose+lstm_1/TensorArrayStack/TensorArrayGatherV3lstm_1/transpose_1/perm*
Tperm0*
T0*4
_output_shapes"
 :������������������
e
lstm_1/Square_1Squarelstm_1/TensorArrayReadV3*
T0*'
_output_shapes
:���������
T
lstm_1/mul_11/xConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
h
lstm_1/mul_11Mullstm_1/mul_11/xlstm_1/Square_1*
T0*'
_output_shapes
:���������
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
valueB"      *
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
valueB
 *����*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
dtype0*
valueB
 *���>*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2�V*
seed���)*
T0*
_output_shapes

:
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:
�
dense_1/kernel
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:
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
�
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
�
dense_1/MatMulMatMullstm_1/TensorArrayReadV3dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������
�
dense_1/BiasAddBiasAdddense_1/MatMuldense_1/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:���������
W
dense_1/TanhTanhdense_1/BiasAdd*
T0*'
_output_shapes
:���������
U
lr/initial_valueConst*
dtype0*
valueB
 *
�#<*
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
�
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
�

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
�
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
�
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
:���������
i
dense_1_targetPlaceholder*
dtype0*
shape: *0
_output_shapes
:������������������
c
subSubdense_1/Tanhdense_1_target*
T0*0
_output_shapes
:������������������
P
SquareSquaresub*
T0*0
_output_shapes
:������������������
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
:���������
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
:���������
X
mulMulMean_1dense_1_sample_weights*
T0*#
_output_shapes
:���������
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
:���������
S
CastCastNotEqual*

DstT0*

SrcT0
*#
_output_shapes
:���������
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
:���������
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
 *  �?*
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
 *  �?*
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
�
gradients/f_count_1Entergradients/f_count*
_output_shapes
: **

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 
�
gradients/MergeMergegradients/f_count_1gradients/NextIteration*
N*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 
�
gradients/SwitchSwitchgradients/Mergelstm_1/while/LoopCond*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 
�
gradients/Add/yConst^lstm_1/while/Identity*
dtype0*&
_class
loc:@lstm_1/while/Exit_1*
value	B :*
_output_shapes
: 
�
gradients/AddAddgradients/Switch:1gradients/Add/y*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
�+
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
�
gradients/b_count_1Entergradients/f_count_2*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 
�
gradients/Merge_1Mergegradients/b_count_1gradients/NextIteration_1*
N*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 
�
gradients/GreaterEqual/EnterEntergradients/b_count*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant(
�
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
�
gradients/Switch_1Switchgradients/Merge_1gradients/b_count_2*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: : 
�
gradients/SubSubgradients/Switch_1:1gradients/GreaterEqual/Enter*&
_class
loc:@lstm_1/while/Exit_1*
T0*
_output_shapes
: 
�
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
�
*gradients/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_1_grad/Shapegradients/add_1_grad/Shape_1*
_class

loc:@add_1*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_1_grad/SumSumgradients/Fill*gradients/add_1_grad/BroadcastGradientArgs*
_class

loc:@add_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_1_grad/ReshapeReshapegradients/add_1_grad/Sumgradients/add_1_grad/Shape*
Tshape0*
_class

loc:@add_1*
T0*
_output_shapes
: 
�
gradients/add_1_grad/Sum_1Sumgradients/Fill,gradients/add_1_grad/BroadcastGradientArgs:1*
_class

loc:@add_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
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
�
(gradients/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/add_grad/Shapegradients/add_grad/Shape_1*
_class

loc:@add*
T0*2
_output_shapes 
:���������:���������
�
gradients/add_grad/SumSumgradients/add_1_grad/Reshape(gradients/add_grad/BroadcastGradientArgs*
_class

loc:@add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/ReshapeReshapegradients/add_grad/Sumgradients/add_grad/Shape*
Tshape0*
_class

loc:@add*
T0*
_output_shapes
: 
�
gradients/add_grad/Sum_1Sumgradients/add_1_grad/Reshape*gradients/add_grad/BroadcastGradientArgs:1*
_class

loc:@add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/add_grad/Reshape_1Reshapegradients/add_grad/Sum_1gradients/add_grad/Shape_1*
Tshape0*
_class

loc:@add*
T0*
_output_shapes
: 
�
!gradients/lstm_1/add_9_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/add_9*
valueB *
_output_shapes
: 
�
#gradients/lstm_1/add_9_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/add_9*
valueB *
_output_shapes
: 
�
1gradients/lstm_1/add_9_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients/lstm_1/add_9_grad/Shape#gradients/lstm_1/add_9_grad/Shape_1*
_class
loc:@lstm_1/add_9*
T0*2
_output_shapes 
:���������:���������
�
gradients/lstm_1/add_9_grad/SumSumgradients/add_1_grad/Reshape_11gradients/lstm_1/add_9_grad/BroadcastGradientArgs*
_class
loc:@lstm_1/add_9*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
#gradients/lstm_1/add_9_grad/ReshapeReshapegradients/lstm_1/add_9_grad/Sum!gradients/lstm_1/add_9_grad/Shape*
Tshape0*
_class
loc:@lstm_1/add_9*
T0*
_output_shapes
: 
�
!gradients/lstm_1/add_9_grad/Sum_1Sumgradients/add_1_grad/Reshape_13gradients/lstm_1/add_9_grad/BroadcastGradientArgs:1*
_class
loc:@lstm_1/add_9*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
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
�
*gradients/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_1_grad/Shapegradients/mul_1_grad/Shape_1*
_class

loc:@mul_1*
T0*2
_output_shapes 
:���������:���������
~
gradients/mul_1_grad/mulMulgradients/add_grad/ReshapeMean_3*
_class

loc:@mul_1*
T0*
_output_shapes
: 
�
gradients/mul_1_grad/SumSumgradients/mul_1_grad/mul*gradients/mul_1_grad/BroadcastGradientArgs*
_class

loc:@mul_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_1_grad/ReshapeReshapegradients/mul_1_grad/Sumgradients/mul_1_grad/Shape*
Tshape0*
_class

loc:@mul_1*
T0*
_output_shapes
: 
�
gradients/mul_1_grad/mul_1Mulmul_1/xgradients/add_grad/Reshape*
_class

loc:@mul_1*
T0*
_output_shapes
: 
�
gradients/mul_1_grad/Sum_1Sumgradients/mul_1_grad/mul_1,gradients/mul_1_grad/BroadcastGradientArgs:1*
_class

loc:@mul_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_1_grad/Reshape_1Reshapegradients/mul_1_grad/Sum_1gradients/mul_1_grad/Shape_1*
Tshape0*
_class

loc:@mul_1*
T0*
_output_shapes
: 
�
gradients/lstm_1/add_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/add*
valueB *
_output_shapes
: 
�
!gradients/lstm_1/add_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/add*
valueB *
_output_shapes
: 
�
/gradients/lstm_1/add_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/lstm_1/add_grad/Shape!gradients/lstm_1/add_grad/Shape_1*
_class
loc:@lstm_1/add*
T0*2
_output_shapes 
:���������:���������
�
gradients/lstm_1/add_grad/SumSumgradients/add_grad/Reshape_1/gradients/lstm_1/add_grad/BroadcastGradientArgs*
_class
loc:@lstm_1/add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
!gradients/lstm_1/add_grad/ReshapeReshapegradients/lstm_1/add_grad/Sumgradients/lstm_1/add_grad/Shape*
Tshape0*
_class
loc:@lstm_1/add*
T0*
_output_shapes
: 
�
gradients/lstm_1/add_grad/Sum_1Sumgradients/add_grad/Reshape_11gradients/lstm_1/add_grad/BroadcastGradientArgs:1*
_class
loc:@lstm_1/add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
#gradients/lstm_1/add_grad/Reshape_1Reshapegradients/lstm_1/add_grad/Sum_1!gradients/lstm_1/add_grad/Shape_1*
Tshape0*
_class
loc:@lstm_1/add*
T0*
_output_shapes
: 
�
)gradients/lstm_1/Sum_2_grad/Reshape/shapeConst*
dtype0*
_class
loc:@lstm_1/Sum_2*
valueB"      *
_output_shapes
:
�
#gradients/lstm_1/Sum_2_grad/ReshapeReshape%gradients/lstm_1/add_9_grad/Reshape_1)gradients/lstm_1/Sum_2_grad/Reshape/shape*
Tshape0*
_class
loc:@lstm_1/Sum_2*
T0*
_output_shapes

:
�
!gradients/lstm_1/Sum_2_grad/ShapeShapelstm_1/mul_11*
_class
loc:@lstm_1/Sum_2*
out_type0*
T0*
_output_shapes
:
�
 gradients/lstm_1/Sum_2_grad/TileTile#gradients/lstm_1/Sum_2_grad/Reshape!gradients/lstm_1/Sum_2_grad/Shape*

Tmultiples0*
_class
loc:@lstm_1/Sum_2*
T0*'
_output_shapes
:���������
�
#gradients/Mean_3_grad/Reshape/shapeConst*
dtype0*
_class
loc:@Mean_3*
valueB:*
_output_shapes
:
�
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
�
gradients/Mean_3_grad/TileTilegradients/Mean_3_grad/Reshapegradients/Mean_3_grad/Shape*

Tmultiples0*
_class
loc:@Mean_3*
T0*#
_output_shapes
:���������
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
�
gradients/Mean_3_grad/ConstConst*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
�
gradients/Mean_3_grad/ProdProdgradients/Mean_3_grad/Shape_1gradients/Mean_3_grad/Const*
_class
loc:@Mean_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
gradients/Mean_3_grad/Const_1Const*
dtype0*
_class
loc:@Mean_3*
valueB: *
_output_shapes
:
�
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
�
gradients/Mean_3_grad/MaximumMaximumgradients/Mean_3_grad/Prod_1gradients/Mean_3_grad/Maximum/y*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
�
gradients/Mean_3_grad/floordivFloorDivgradients/Mean_3_grad/Prodgradients/Mean_3_grad/Maximum*
_class
loc:@Mean_3*
T0*
_output_shapes
: 
�
gradients/Mean_3_grad/CastCastgradients/Mean_3_grad/floordiv*

DstT0*
_class
loc:@Mean_3*

SrcT0*
_output_shapes
: 
�
gradients/Mean_3_grad/truedivRealDivgradients/Mean_3_grad/Tilegradients/Mean_3_grad/Cast*
_class
loc:@Mean_3*
T0*#
_output_shapes
:���������
�
'gradients/lstm_1/Sum_grad/Reshape/shapeConst*
dtype0*
_class
loc:@lstm_1/Sum*
valueB"      *
_output_shapes
:
�
!gradients/lstm_1/Sum_grad/ReshapeReshape#gradients/lstm_1/add_grad/Reshape_1'gradients/lstm_1/Sum_grad/Reshape/shape*
Tshape0*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes

:
�
(gradients/lstm_1/Sum_grad/Tile/multiplesConst*
dtype0*
_class
loc:@lstm_1/Sum*
valueB"   `   *
_output_shapes
:
�
gradients/lstm_1/Sum_grad/TileTile!gradients/lstm_1/Sum_grad/Reshape(gradients/lstm_1/Sum_grad/Tile/multiples*

Tmultiples0*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes

:`
�
"gradients/lstm_1/mul_11_grad/ShapeConst*
dtype0* 
_class
loc:@lstm_1/mul_11*
valueB *
_output_shapes
: 
�
$gradients/lstm_1/mul_11_grad/Shape_1Shapelstm_1/Square_1* 
_class
loc:@lstm_1/mul_11*
out_type0*
T0*
_output_shapes
:
�
2gradients/lstm_1/mul_11_grad/BroadcastGradientArgsBroadcastGradientArgs"gradients/lstm_1/mul_11_grad/Shape$gradients/lstm_1/mul_11_grad/Shape_1* 
_class
loc:@lstm_1/mul_11*
T0*2
_output_shapes 
:���������:���������
�
 gradients/lstm_1/mul_11_grad/mulMul gradients/lstm_1/Sum_2_grad/Tilelstm_1/Square_1* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:���������
�
 gradients/lstm_1/mul_11_grad/SumSum gradients/lstm_1/mul_11_grad/mul2gradients/lstm_1/mul_11_grad/BroadcastGradientArgs* 
_class
loc:@lstm_1/mul_11*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
$gradients/lstm_1/mul_11_grad/ReshapeReshape gradients/lstm_1/mul_11_grad/Sum"gradients/lstm_1/mul_11_grad/Shape*
Tshape0* 
_class
loc:@lstm_1/mul_11*
T0*
_output_shapes
: 
�
"gradients/lstm_1/mul_11_grad/mul_1Mullstm_1/mul_11/x gradients/lstm_1/Sum_2_grad/Tile* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:���������
�
"gradients/lstm_1/mul_11_grad/Sum_1Sum"gradients/lstm_1/mul_11_grad/mul_14gradients/lstm_1/mul_11_grad/BroadcastGradientArgs:1* 
_class
loc:@lstm_1/mul_11*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
&gradients/lstm_1/mul_11_grad/Reshape_1Reshape"gradients/lstm_1/mul_11_grad/Sum_1$gradients/lstm_1/mul_11_grad/Shape_1*
Tshape0* 
_class
loc:@lstm_1/mul_11*
T0*'
_output_shapes
:���������
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
�
(gradients/div_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/div_grad/Shapegradients/div_grad/Shape_1*
_class

loc:@div*
T0*2
_output_shapes 
:���������:���������
�
gradients/div_grad/RealDivRealDivgradients/Mean_3_grad/truedivMean_2*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/SumSumgradients/div_grad/RealDiv(gradients/div_grad/BroadcastGradientArgs*
_class

loc:@div*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/div_grad/ReshapeReshapegradients/div_grad/Sumgradients/div_grad/Shape*
Tshape0*
_class

loc:@div*
T0*#
_output_shapes
:���������
h
gradients/div_grad/NegNegmul*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/RealDiv_1RealDivgradients/div_grad/NegMean_2*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/RealDiv_2RealDivgradients/div_grad/RealDiv_1Mean_2*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/mulMulgradients/Mean_3_grad/truedivgradients/div_grad/RealDiv_2*
_class

loc:@div*
T0*#
_output_shapes
:���������
�
gradients/div_grad/Sum_1Sumgradients/div_grad/mul*gradients/div_grad/BroadcastGradientArgs:1*
_class

loc:@div*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/div_grad/Reshape_1Reshapegradients/div_grad/Sum_1gradients/div_grad/Shape_1*
Tshape0*
_class

loc:@div*
T0*
_output_shapes
: 
�
gradients/lstm_1/mul_grad/ShapeConst*
dtype0*
_class
loc:@lstm_1/mul*
valueB *
_output_shapes
: 
�
!gradients/lstm_1/mul_grad/Shape_1Const*
dtype0*
_class
loc:@lstm_1/mul*
valueB"   `   *
_output_shapes
:
�
/gradients/lstm_1/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/lstm_1/mul_grad/Shape!gradients/lstm_1/mul_grad/Shape_1*
_class
loc:@lstm_1/mul*
T0*2
_output_shapes 
:���������:���������
�
gradients/lstm_1/mul_grad/mulMulgradients/lstm_1/Sum_grad/Tilelstm_1/Square*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:`
�
gradients/lstm_1/mul_grad/SumSumgradients/lstm_1/mul_grad/mul/gradients/lstm_1/mul_grad/BroadcastGradientArgs*
_class
loc:@lstm_1/mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
!gradients/lstm_1/mul_grad/ReshapeReshapegradients/lstm_1/mul_grad/Sumgradients/lstm_1/mul_grad/Shape*
Tshape0*
_class
loc:@lstm_1/mul*
T0*
_output_shapes
: 
�
gradients/lstm_1/mul_grad/mul_1Mullstm_1/mul/xgradients/lstm_1/Sum_grad/Tile*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:`
�
gradients/lstm_1/mul_grad/Sum_1Sumgradients/lstm_1/mul_grad/mul_11gradients/lstm_1/mul_grad/BroadcastGradientArgs:1*
_class
loc:@lstm_1/mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
#gradients/lstm_1/mul_grad/Reshape_1Reshapegradients/lstm_1/mul_grad/Sum_1!gradients/lstm_1/mul_grad/Shape_1*
Tshape0*
_class
loc:@lstm_1/mul*
T0*
_output_shapes

:`
�
$gradients/lstm_1/Square_1_grad/mul/xConst'^gradients/lstm_1/mul_11_grad/Reshape_1*
dtype0*"
_class
loc:@lstm_1/Square_1*
valueB
 *   @*
_output_shapes
: 
�
"gradients/lstm_1/Square_1_grad/mulMul$gradients/lstm_1/Square_1_grad/mul/xlstm_1/TensorArrayReadV3*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������
�
$gradients/lstm_1/Square_1_grad/mul_1Mul&gradients/lstm_1/mul_11_grad/Reshape_1"gradients/lstm_1/Square_1_grad/mul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������
v
gradients/mul_grad/ShapeShapeMean_1*
_class

loc:@mul*
out_type0*
T0*
_output_shapes
:
�
gradients/mul_grad/Shape_1Shapedense_1_sample_weights*
_class

loc:@mul*
out_type0*
T0*
_output_shapes
:
�
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*
_class

loc:@mul*
T0*2
_output_shapes 
:���������:���������
�
gradients/mul_grad/mulMulgradients/div_grad/Reshapedense_1_sample_weights*
_class

loc:@mul*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/SumSumgradients/mul_grad/mul(gradients/mul_grad/BroadcastGradientArgs*
_class

loc:@mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
Tshape0*
_class

loc:@mul*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/mul_1MulMean_1gradients/div_grad/Reshape*
_class

loc:@mul*
T0*#
_output_shapes
:���������
�
gradients/mul_grad/Sum_1Sumgradients/mul_grad/mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
_class

loc:@mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
Tshape0*
_class

loc:@mul*
T0*#
_output_shapes
:���������
�
"gradients/lstm_1/Square_grad/mul/xConst$^gradients/lstm_1/mul_grad/Reshape_1*
dtype0* 
_class
loc:@lstm_1/Square*
valueB
 *   @*
_output_shapes
: 
�
 gradients/lstm_1/Square_grad/mulMul"gradients/lstm_1/Square_grad/mul/xlstm_1/kernel/read* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:`
�
"gradients/lstm_1/Square_grad/mul_1Mul#gradients/lstm_1/mul_grad/Reshape_1 gradients/lstm_1/Square_grad/mul* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:`
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
�
gradients/Mean_1_grad/addAddMean_1/reduction_indicesgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
gradients/Mean_1_grad/modFloorModgradients/Mean_1_grad/addgradients/Mean_1_grad/Size*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
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
�
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
�
gradients/Mean_1_grad/FillFillgradients/Mean_1_grad/Shape_1 gradients/Mean_1_grad/Fill/value*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
#gradients/Mean_1_grad/DynamicStitchDynamicStitchgradients/Mean_1_grad/rangegradients/Mean_1_grad/modgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Fill*
N*
_class
loc:@Mean_1*
T0*#
_output_shapes
:���������
|
gradients/Mean_1_grad/Maximum/yConst*
dtype0*
_class
loc:@Mean_1*
value	B :*
_output_shapes
: 
�
gradients/Mean_1_grad/MaximumMaximum#gradients/Mean_1_grad/DynamicStitchgradients/Mean_1_grad/Maximum/y*
_class
loc:@Mean_1*
T0*#
_output_shapes
:���������
�
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Shapegradients/Mean_1_grad/Maximum*
_class
loc:@Mean_1*
T0*#
_output_shapes
:���������
�
gradients/Mean_1_grad/ReshapeReshapegradients/mul_grad/Reshape#gradients/Mean_1_grad/DynamicStitch*
Tshape0*
_class
loc:@Mean_1*
T0*
_output_shapes
:
�
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
�
gradients/Mean_1_grad/ConstConst*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
�
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const*
_class
loc:@Mean_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
�
gradients/Mean_1_grad/Const_1Const*
dtype0*
_class
loc:@Mean_1*
valueB: *
_output_shapes
:
�
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
�
gradients/Mean_1_grad/Maximum_1Maximumgradients/Mean_1_grad/Prod_1!gradients/Mean_1_grad/Maximum_1/y*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
 gradients/Mean_1_grad/floordiv_1FloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum_1*
_class
loc:@Mean_1*
T0*
_output_shapes
: 
�
gradients/Mean_1_grad/CastCast gradients/Mean_1_grad/floordiv_1*

DstT0*
_class
loc:@Mean_1*

SrcT0*
_output_shapes
: 
�
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*
_class
loc:@Mean_1*
T0*#
_output_shapes
:���������
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
�
gradients/Mean_grad/addAddMean/reduction_indicesgradients/Mean_grad/Size*
_class
	loc:@Mean*
T0*
_output_shapes
: 
�
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
�
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
�
gradients/Mean_grad/FillFillgradients/Mean_grad/Shape_1gradients/Mean_grad/Fill/value*
_class
	loc:@Mean*
T0*
_output_shapes
: 
�
!gradients/Mean_grad/DynamicStitchDynamicStitchgradients/Mean_grad/rangegradients/Mean_grad/modgradients/Mean_grad/Shapegradients/Mean_grad/Fill*
N*
_class
	loc:@Mean*
T0*#
_output_shapes
:���������
x
gradients/Mean_grad/Maximum/yConst*
dtype0*
_class
	loc:@Mean*
value	B :*
_output_shapes
: 
�
gradients/Mean_grad/MaximumMaximum!gradients/Mean_grad/DynamicStitchgradients/Mean_grad/Maximum/y*
_class
	loc:@Mean*
T0*#
_output_shapes
:���������
�
gradients/Mean_grad/floordivFloorDivgradients/Mean_grad/Shapegradients/Mean_grad/Maximum*
_class
	loc:@Mean*
T0*
_output_shapes
:
�
gradients/Mean_grad/ReshapeReshapegradients/Mean_1_grad/truediv!gradients/Mean_grad/DynamicStitch*
Tshape0*
_class
	loc:@Mean*
T0*
_output_shapes
:
�
gradients/Mean_grad/TileTilegradients/Mean_grad/Reshapegradients/Mean_grad/floordiv*

Tmultiples0*
_class
	loc:@Mean*
T0*0
_output_shapes
:������������������
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
�
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
�
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
�
gradients/Mean_grad/Maximum_1Maximumgradients/Mean_grad/Prod_1gradients/Mean_grad/Maximum_1/y*
_class
	loc:@Mean*
T0*
_output_shapes
: 
�
gradients/Mean_grad/floordiv_1FloorDivgradients/Mean_grad/Prodgradients/Mean_grad/Maximum_1*
_class
	loc:@Mean*
T0*
_output_shapes
: 
�
gradients/Mean_grad/CastCastgradients/Mean_grad/floordiv_1*

DstT0*
_class
	loc:@Mean*

SrcT0*
_output_shapes
: 
�
gradients/Mean_grad/truedivRealDivgradients/Mean_grad/Tilegradients/Mean_grad/Cast*
_class
	loc:@Mean*
T0*0
_output_shapes
:������������������
�
gradients/Square_grad/mul/xConst^gradients/Mean_grad/truediv*
dtype0*
_class
loc:@Square*
valueB
 *   @*
_output_shapes
: 
�
gradients/Square_grad/mulMulgradients/Square_grad/mul/xsub*
_class
loc:@Square*
T0*0
_output_shapes
:������������������
�
gradients/Square_grad/mul_1Mulgradients/Mean_grad/truedivgradients/Square_grad/mul*
_class
loc:@Square*
T0*0
_output_shapes
:������������������
|
gradients/sub_grad/ShapeShapedense_1/Tanh*
_class

loc:@sub*
out_type0*
T0*
_output_shapes
:
�
gradients/sub_grad/Shape_1Shapedense_1_target*
_class

loc:@sub*
out_type0*
T0*
_output_shapes
:
�
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*
_class

loc:@sub*
T0*2
_output_shapes 
:���������:���������
�
gradients/sub_grad/SumSumgradients/Square_grad/mul_1(gradients/sub_grad/BroadcastGradientArgs*
_class

loc:@sub*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
Tshape0*
_class

loc:@sub*
T0*'
_output_shapes
:���������
�
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
�
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
Tshape0*
_class

loc:@sub*
T0*0
_output_shapes
:������������������
�
$gradients/dense_1/Tanh_grad/TanhGradTanhGraddense_1/Tanhgradients/sub_grad/Reshape*
_class
loc:@dense_1/Tanh*
T0*'
_output_shapes
:���������
�
*gradients/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad$gradients/dense_1/Tanh_grad/TanhGrad*"
_class
loc:@dense_1/BiasAdd*
data_formatNHWC*
T0*
_output_shapes
:
�
$gradients/dense_1/MatMul_grad/MatMulMatMul$gradients/dense_1/Tanh_grad/TanhGraddense_1/kernel/read*
transpose_b(*
transpose_a( *!
_class
loc:@dense_1/MatMul*
T0*'
_output_shapes
:���������
�
&gradients/dense_1/MatMul_grad/MatMul_1MatMullstm_1/TensorArrayReadV3$gradients/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_1/MatMul*
T0*
_output_shapes

:
�
gradients/AddNAddN$gradients/lstm_1/Square_1_grad/mul_1$gradients/dense_1/MatMul_grad/MatMul*
N*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������
�
Igradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArraylstm_1/while/Exit_1*
source	gradients*%
_class
loc:@lstm_1/TensorArray*
_output_shapes

::
�
Egradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentitylstm_1/while/Exit_1J^gradients/lstm_1/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
:
�
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
:���������
j
gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_3*
T0*'
_output_shapes
:���������
�
)gradients/lstm_1/while/Exit_1_grad/b_exitEnterKgradients/lstm_1/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_1*
is_constant( 
�
)gradients/lstm_1/while/Exit_2_grad/b_exitEntergradients/zeros_like*'
_output_shapes
:���������*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_2*
is_constant( 
�
)gradients/lstm_1/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*'
_output_shapes
:���������*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_3*
is_constant( 
�
-gradients/lstm_1/while/Switch_1_grad/b_switchMerge)gradients/lstm_1/while/Exit_1_grad/b_exit4gradients/lstm_1/while/Switch_1_grad_1/NextIteration*
N*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: : 
�
-gradients/lstm_1/while/Switch_2_grad/b_switchMerge)gradients/lstm_1/while/Exit_2_grad/b_exit4gradients/lstm_1/while/Switch_2_grad_1/NextIteration*
N*'
_class
loc:@lstm_1/while/Merge_2*
T0*)
_output_shapes
:���������: 
�
-gradients/lstm_1/while/Switch_3_grad/b_switchMerge)gradients/lstm_1/while/Exit_3_grad/b_exit4gradients/lstm_1/while/Switch_3_grad_1/NextIteration*
N*'
_class
loc:@lstm_1/while/Merge_3*
T0*)
_output_shapes
:���������: 
�
*gradients/lstm_1/while/Merge_1_grad/SwitchSwitch-gradients/lstm_1/while/Switch_1_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: : 
�
*gradients/lstm_1/while/Merge_2_grad/SwitchSwitch-gradients/lstm_1/while/Switch_2_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_2*
T0*:
_output_shapes(
&:���������:���������
�
*gradients/lstm_1/while/Merge_3_grad/SwitchSwitch-gradients/lstm_1/while/Switch_3_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:���������:���������
�
(gradients/lstm_1/while/Enter_1_grad/ExitExit*gradients/lstm_1/while/Merge_1_grad/Switch*'
_class
loc:@lstm_1/while/Enter_1*
T0*
_output_shapes
: 
�
ggradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/TensorArray*
is_constant(
�
agradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3ggradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter,gradients/lstm_1/while/Merge_1_grad/Switch:1*
source	gradients*%
_class
loc:@lstm_1/TensorArray*
_output_shapes

::
�
]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flowIdentity,gradients/lstm_1/while/Merge_1_grad/Switch:1b^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3*%
_class
loc:@lstm_1/TensorArray*
T0*
_output_shapes
: 
�
Wgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
	elem_type0*
_output_shapes
:
�
Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnterRefEnterWgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
is_constant(
�
[gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPush	StackPushZgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnterlstm_1/while/Identity^gradients/Add*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
T0*
swap_memory(*
_output_shapes
:
�
cgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop/RefEnterRefEnterWgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
is_constant(
�
Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopStackPopcgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/TensorArrayloc:@lstm_1/while/Identity*
	elem_type0*
_output_shapes
: 
�+
Xgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/b_syncControlTrigger[^gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopA^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_9_grad/mul/StackPop1^gradients/lstm_1/while/mul_9_grad/mul_1/StackPopB^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopD^gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1K^gradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopG^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopI^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1S^gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1A^gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_4_grad/mul/StackPop1^gradients/lstm_1/while/mul_4_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/^gradients/lstm_1/while/mul_6_grad/mul/StackPop1^gradients/lstm_1/while/mul_6_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_8_grad/mul/StackPop1^gradients/lstm_1/while/mul_8_grad/mul_1/StackPopB^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopD^gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1K^gradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop@^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopB^gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1I^gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1G^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopI^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1S^gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopE^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopG^gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1Q^gradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1F^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopA^gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopF^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPopA^gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_7_grad/mul/StackPop1^gradients/lstm_1/while/mul_7_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_3_grad/mul/StackPop1^gradients/lstm_1/while/mul_3_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_1_grad/mul/StackPop1^gradients/lstm_1/while/mul_1_grad/mul_1/StackPopA^gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_5_grad/mul/StackPopA^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopC^gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1?^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPopA^gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1F^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopH^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1H^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2H^gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_37^gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPopD^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopF^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1F^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2F^gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_35^gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPopA^gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/^gradients/lstm_1/while/mul_2_grad/mul/StackPop?^gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop-^gradients/lstm_1/while/mul_grad/mul/StackPop*%
_class
loc:@lstm_1/TensorArray
�
Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3TensorArrayReadV3agradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/TensorArrayGradV3Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPop]gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayGrad/gradient_flow*
dtype0*%
_class
loc:@lstm_1/TensorArray*'
_output_shapes
:���������
�
gradients/AddN_1AddN,gradients/lstm_1/while/Merge_2_grad/Switch:1Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
N*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_9_grad/ShapeShapelstm_1/while/clip_by_value_2*%
_class
loc:@lstm_1/while/mul_9*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/mul_9_grad/Shape_1Shapelstm_1/while/Tanh_1*%
_class
loc:@lstm_1/while/mul_9*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
�
Agradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_9_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
�
@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
�
Cgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_9_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_9*
is_constant(
�
Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_9*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_9_grad/mul/f_accStack*

stack_name *?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_9_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_9_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
is_constant(
�
/gradients/lstm_1/while/mul_9_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_9_grad/mul/RefEnterlstm_1/while/Tanh_1^gradients/Add*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_9_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_9_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
is_constant(
�
.gradients/lstm_1/while/mul_9_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_9_grad/mul/StackPop/RefEnter^gradients/Sub*?
_class5
3loc:@lstm_1/while/Tanh_1loc:@lstm_1/while/mul_9*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_9_grad/mulMulgradients/AddN_1.gradients/lstm_1/while/mul_9_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_9_grad/SumSum%gradients/lstm_1/while/mul_9_grad/mul7gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_9*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_9_grad/ReshapeReshape%gradients/lstm_1/while/mul_9_grad/Sum@gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/mul_9_grad/mul_1/f_accStack*

stack_name *H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_9_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_9_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
is_constant(
�
1gradients/lstm_1/while/mul_9_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_9_grad/mul_1/RefEnterlstm_1/while/clip_by_value_2^gradients/Add*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_9_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_9_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
is_constant(
�
0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_9_grad/mul_1/StackPop/RefEnter^gradients/Sub*H
_class>
<!loc:@lstm_1/while/clip_by_value_2loc:@lstm_1/while/mul_9*
	elem_type0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_9_grad/mul_1Mul0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopgradients/AddN_1*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_9_grad/Sum_1Sum'gradients/lstm_1/while/mul_9_grad/mul_19gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_9*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_9_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_9_grad/Sum_1Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������
�
1gradients/lstm_1/while/clip_by_value_2_grad/ShapeShape$lstm_1/while/clip_by_value_2/Minimum*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
T0*
_output_shapes
:
�
3gradients/lstm_1/while/clip_by_value_2_grad/Shape_1Const^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB *
_output_shapes
: 
�
3gradients/lstm_1/while/clip_by_value_2_grad/Shape_2Shape)gradients/lstm_1/while/mul_9_grad/Reshape*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
out_type0*
T0*
_output_shapes
:
�
7gradients/lstm_1/while/clip_by_value_2_grad/zeros/ConstConst^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
valueB
 *    *
_output_shapes
: 
�
1gradients/lstm_1/while/clip_by_value_2_grad/zerosFill3gradients/lstm_1/while/clip_by_value_2_grad/Shape_27gradients/lstm_1/while/clip_by_value_2_grad/zeros/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_accStack*

stack_name *Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Bgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush	StackPushAgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter$lstm_1/while/clip_by_value_2/Minimum^gradients/Add*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Agradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopStackPopJgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/clip_by_value_2)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*'
_output_shapes
:���������
�
@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1Stack*

stack_name *J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
�
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1RefEnter@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
is_constant(
�
Dgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPush_1	StackPushCgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1lstm_1/while/Const_4^gradients/Add*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
T0*
swap_memory(*
_output_shapes
:
�
Lgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1/RefEnterRefEnter@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
is_constant(
�
Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1StackPopLgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*J
_class@
>loc:@lstm_1/while/Const_4!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
: 
�
8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqualGreaterEqualAgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPopCgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/StackPop_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
Ggradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
is_constant(
�
Kgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPush	StackPushJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnter1gradients/lstm_1/while/clip_by_value_2_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
swap_memory(*
_output_shapes
:
�
Sgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
is_constant(
�
Jgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPopStackPopSgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop3gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*2
_output_shapes 
:���������:���������
�
2gradients/lstm_1/while/clip_by_value_2_grad/SelectSelect8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:���������
�
4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
/gradients/lstm_1/while/clip_by_value_2_grad/SumSum2gradients/lstm_1/while/clip_by_value_2_grad/SelectAgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
3gradients/lstm_1/while/clip_by_value_2_grad/ReshapeReshape/gradients/lstm_1/while/clip_by_value_2_grad/SumJgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/StackPop*
Tshape0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������
�
1gradients/lstm_1/while/clip_by_value_2_grad/Sum_1Sum4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Cgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
5gradients/lstm_1/while/clip_by_value_2_grad/Reshape_1Reshape1gradients/lstm_1/while/clip_by_value_2_grad/Sum_13gradients/lstm_1/while/clip_by_value_2_grad/Shape_1*
Tshape0*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*
_output_shapes
: 
�
+gradients/lstm_1/while/Tanh_1_grad/TanhGradTanhGrad.gradients/lstm_1/while/mul_9_grad/mul/StackPop+gradients/lstm_1/while/mul_9_grad/Reshape_1*&
_class
loc:@lstm_1/while/Tanh_1*
T0*'
_output_shapes
:���������
�
4gradients/lstm_1/while/Switch_1_grad_1/NextIterationNextIteration,gradients/lstm_1/while/Merge_1_grad/Switch:1*'
_class
loc:@lstm_1/while/Merge_1*
T0*
_output_shapes
: 
�
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ShapeShapelstm_1/while/add_7*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
T0*
_output_shapes
:
�
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB *
_output_shapes
: 
�
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2Shape3gradients/lstm_1/while/clip_by_value_2_grad/Reshape*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
out_type0*
T0*
_output_shapes
:
�
?gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
valueB
 *    *
_output_shapes
: 
�
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zerosFill;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_2?gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
Cgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_accStack*

stack_name *P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Fgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Ggradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush	StackPushFgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_7^gradients/Add*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Ogradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Fgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopStackPopOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/add_7)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*'
_output_shapes
:���������
�
Egradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1RefEnterEgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPush_1	StackPushHgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_5^gradients/Add*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Qgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1StackPopQgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*R
_classH
Floc:@lstm_1/while/Const_5)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
: 
�
=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual	LessEqualFgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPopHgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/StackPop_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
Ogradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Rgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Sgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnter9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape^gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
[gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
is_constant(
�
Rgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPopStackPop[gradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
	elem_type0*
_output_shapes
:
�
Igradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*2
_output_shapes 
:���������:���������
�
:gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectSelect=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:���������
�
<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
7gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumSum:gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SelectIgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/ReshapeReshape7gradients/lstm_1/while/clip_by_value_2/Minimum_grad/SumRgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/StackPop*
Tshape0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������
�
9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1Sum<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Kgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs:1*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape_1Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Sum_1;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Shape_1*
Tshape0*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*
_output_shapes
: 
�
gradients/AddN_2AddN,gradients/lstm_1/while/Merge_3_grad/Switch:1+gradients/lstm_1/while/Tanh_1_grad/TanhGrad*
N*'
_class
loc:@lstm_1/while/Merge_3*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_5_grad/ShapeShapelstm_1/while/mul_4*%
_class
loc:@lstm_1/while/add_5*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/add_5_grad/Shape_1Shapelstm_1/while/mul_6*%
_class
loc:@lstm_1/while/add_5*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
�
Agradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_5_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_5*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
�
@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
�
Cgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_5_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_5*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_5*
is_constant(
�
Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_5*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_5_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_5*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_5_grad/SumSumgradients/AddN_27gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/add_5_grad/ReshapeReshape%gradients/lstm_1/while/add_5_grad/Sum@gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_5*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_5_grad/Sum_1Sumgradients/AddN_29gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/add_5_grad/Reshape_1Reshape'gradients/lstm_1/while/add_5_grad/Sum_1Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/add_5*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_7_grad/ShapeShapelstm_1/while/mul_8*%
_class
loc:@lstm_1/while/add_7*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/add_7_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_7*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_7*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_7*
is_constant(
�
Agradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_7_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_7*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_7*
is_constant(
�
@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_7*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_7_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_7_grad/Shape_1*%
_class
loc:@lstm_1/while/add_7*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_7_grad/SumSum;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape7gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/add_7_grad/ReshapeReshape%gradients/lstm_1/while/add_7_grad/Sum@gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_7*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_7_grad/Sum_1Sum;gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Reshape9gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/add_7_grad/Reshape_1Reshape'gradients/lstm_1/while/add_7_grad/Sum_1)gradients/lstm_1/while/add_7_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/add_7*
T0*
_output_shapes
: 
�
'gradients/lstm_1/while/mul_4_grad/ShapeShapelstm_1/while/clip_by_value_1*%
_class
loc:@lstm_1/while/mul_4*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/mul_4_grad/Shape_1Shapelstm_1/while/Identity_3*%
_class
loc:@lstm_1/while/mul_4*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
�
Agradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_4_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
�
@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
�
Cgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_4_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_4*
is_constant(
�
Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_4*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_4_grad/mul/f_accStack*

stack_name *C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_4_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_4_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
is_constant(
�
/gradients/lstm_1/while/mul_4_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_4_grad/mul/RefEnterlstm_1/while/Identity_3^gradients/Add*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_4_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_4_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
is_constant(
�
.gradients/lstm_1/while/mul_4_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_4_grad/mul/StackPop/RefEnter^gradients/Sub*C
_class9
7loc:@lstm_1/while/Identity_3loc:@lstm_1/while/mul_4*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_4_grad/mulMul)gradients/lstm_1/while/add_5_grad/Reshape.gradients/lstm_1/while/mul_4_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_4_grad/SumSum%gradients/lstm_1/while/mul_4_grad/mul7gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_4_grad/ReshapeReshape%gradients/lstm_1/while/mul_4_grad/Sum@gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/mul_4_grad/mul_1/f_accStack*

stack_name *H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_4_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_4_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
is_constant(
�
1gradients/lstm_1/while/mul_4_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_4_grad/mul_1/RefEnterlstm_1/while/clip_by_value_1^gradients/Add*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_4_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_4_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
is_constant(
�
0gradients/lstm_1/while/mul_4_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_4_grad/mul_1/StackPop/RefEnter^gradients/Sub*H
_class>
<!loc:@lstm_1/while/clip_by_value_1loc:@lstm_1/while/mul_4*
	elem_type0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_4_grad/mul_1Mul0gradients/lstm_1/while/mul_4_grad/mul_1/StackPop)gradients/lstm_1/while/add_5_grad/Reshape*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_4_grad/Sum_1Sum'gradients/lstm_1/while/mul_4_grad/mul_19gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_4_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_4_grad/Sum_1Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_6_grad/ShapeShapelstm_1/while/clip_by_value*%
_class
loc:@lstm_1/while/mul_6*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/mul_6_grad/Shape_1Shapelstm_1/while/Tanh*%
_class
loc:@lstm_1/while/mul_6*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
�
Agradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_6_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
�
@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
�
Cgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/mul_6_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_6*
is_constant(
�
Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/mul_6*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_6_grad/mul/f_accStack*

stack_name *=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_6_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_6_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
is_constant(
�
/gradients/lstm_1/while/mul_6_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_6_grad/mul/RefEnterlstm_1/while/Tanh^gradients/Add*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_6_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_6_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
is_constant(
�
.gradients/lstm_1/while/mul_6_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_6_grad/mul/StackPop/RefEnter^gradients/Sub*=
_class3
1loc:@lstm_1/while/Tanhloc:@lstm_1/while/mul_6*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_6_grad/mulMul+gradients/lstm_1/while/add_5_grad/Reshape_1.gradients/lstm_1/while/mul_6_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_6_grad/SumSum%gradients/lstm_1/while/mul_6_grad/mul7gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_6*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_6_grad/ReshapeReshape%gradients/lstm_1/while/mul_6_grad/Sum@gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/mul_6_grad/mul_1/f_accStack*

stack_name *F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_6_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_6_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
is_constant(
�
1gradients/lstm_1/while/mul_6_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_6_grad/mul_1/RefEnterlstm_1/while/clip_by_value^gradients/Add*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_6_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_6_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
is_constant(
�
0gradients/lstm_1/while/mul_6_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_6_grad/mul_1/StackPop/RefEnter^gradients/Sub*F
_class<
:loc:@lstm_1/while/clip_by_valueloc:@lstm_1/while/mul_6*
	elem_type0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_6_grad/mul_1Mul0gradients/lstm_1/while/mul_6_grad/mul_1/StackPop+gradients/lstm_1/while/add_5_grad/Reshape_1*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_6_grad/Sum_1Sum'gradients/lstm_1/while/mul_6_grad/mul_19gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_6*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_6_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_6_grad/Sum_1Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_8_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_8*
valueB *
_output_shapes
: 
�
)gradients/lstm_1/while/mul_8_grad/Shape_1Shapelstm_1/while/add_6*%
_class
loc:@lstm_1/while/mul_8*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_8*
is_constant(
�
Agradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_8_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_8*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_8*
is_constant(
�
@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_8_grad/Shape@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_8_grad/mul/f_accStack*

stack_name *>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_8_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_8_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
is_constant(
�
/gradients/lstm_1/while/mul_8_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_8_grad/mul/RefEnterlstm_1/while/add_6^gradients/Add*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_8_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_8_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
is_constant(
�
.gradients/lstm_1/while/mul_8_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_8_grad/mul/StackPop/RefEnter^gradients/Sub*>
_class4
2loc:@lstm_1/while/add_6loc:@lstm_1/while/mul_8*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_8_grad/mulMul)gradients/lstm_1/while/add_7_grad/Reshape.gradients/lstm_1/while/mul_8_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_8_grad/SumSum%gradients/lstm_1/while/mul_8_grad/mul7gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_8*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_8_grad/ReshapeReshape%gradients/lstm_1/while/mul_8_grad/Sum'gradients/lstm_1/while/mul_8_grad/Shape*
Tshape0*%
_class
loc:@lstm_1/while/mul_8*
T0*
_output_shapes
: 
�
-gradients/lstm_1/while/mul_8_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_8_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_8_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
is_constant(
�
1gradients/lstm_1/while/mul_8_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_8_grad/mul_1/RefEnterlstm_1/while/mul_8/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_8_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_8_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
is_constant(
�
0gradients/lstm_1/while/mul_8_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_8_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_8loc:@lstm_1/while/mul_8/x*
	elem_type0*
_output_shapes
: 
�
'gradients/lstm_1/while/mul_8_grad/mul_1Mul0gradients/lstm_1/while/mul_8_grad/mul_1/StackPop)gradients/lstm_1/while/add_7_grad/Reshape*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_8_grad/Sum_1Sum'gradients/lstm_1/while/mul_8_grad/mul_19gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_8*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_8_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_8_grad/Sum_1@gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:���������
�
1gradients/lstm_1/while/clip_by_value_1_grad/ShapeShape$lstm_1/while/clip_by_value_1/Minimum*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
T0*
_output_shapes
:
�
3gradients/lstm_1/while/clip_by_value_1_grad/Shape_1Const^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB *
_output_shapes
: 
�
3gradients/lstm_1/while/clip_by_value_1_grad/Shape_2Shape)gradients/lstm_1/while/mul_4_grad/Reshape*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
out_type0*
T0*
_output_shapes
:
�
7gradients/lstm_1/while/clip_by_value_1_grad/zeros/ConstConst^gradients/Sub*
dtype0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
valueB
 *    *
_output_shapes
: 
�
1gradients/lstm_1/while/clip_by_value_1_grad/zerosFill3gradients/lstm_1/while/clip_by_value_1_grad/Shape_27gradients/lstm_1/while/clip_by_value_1_grad/zeros/Const*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_accStack*

stack_name *Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Bgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush	StackPushAgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter$lstm_1/while/clip_by_value_1/Minimum^gradients/Add*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Agradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopStackPopJgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/clip_by_value_1)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*'
_output_shapes
:���������
�
@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1Stack*

stack_name *J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
�
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1RefEnter@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
is_constant(
�
Dgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPush_1	StackPushCgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1lstm_1/while/Const_2^gradients/Add*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
T0*
swap_memory(*
_output_shapes
:
�
Lgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1/RefEnterRefEnter@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
is_constant(
�
Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1StackPopLgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*J
_class@
>loc:@lstm_1/while/Const_2!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
: 
�
8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqualGreaterEqualAgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPopCgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/StackPop_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
Ggradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
is_constant(
�
Kgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPush	StackPushJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnter1gradients/lstm_1/while/clip_by_value_1_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
swap_memory(*
_output_shapes
:
�
Sgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterGgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
is_constant(
�
Jgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPopStackPopSgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgsBroadcastGradientArgsJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop3gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*2
_output_shapes 
:���������:���������
�
2gradients/lstm_1/while/clip_by_value_1_grad/SelectSelect8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:���������
�
4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
/gradients/lstm_1/while/clip_by_value_1_grad/SumSum2gradients/lstm_1/while/clip_by_value_1_grad/SelectAgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
3gradients/lstm_1/while/clip_by_value_1_grad/ReshapeReshape/gradients/lstm_1/while/clip_by_value_1_grad/SumJgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/StackPop*
Tshape0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������
�
1gradients/lstm_1/while/clip_by_value_1_grad/Sum_1Sum4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Cgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs:1*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
5gradients/lstm_1/while/clip_by_value_1_grad/Reshape_1Reshape1gradients/lstm_1/while/clip_by_value_1_grad/Sum_13gradients/lstm_1/while/clip_by_value_1_grad/Shape_1*
Tshape0*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*
_output_shapes
: 
�
/gradients/lstm_1/while/clip_by_value_grad/ShapeShape"lstm_1/while/clip_by_value/Minimum*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
T0*
_output_shapes
:
�
1gradients/lstm_1/while/clip_by_value_grad/Shape_1Const^gradients/Sub*
dtype0*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB *
_output_shapes
: 
�
1gradients/lstm_1/while/clip_by_value_grad/Shape_2Shape)gradients/lstm_1/while/mul_6_grad/Reshape*-
_class#
!loc:@lstm_1/while/clip_by_value*
out_type0*
T0*
_output_shapes
:
�
5gradients/lstm_1/while/clip_by_value_grad/zeros/ConstConst^gradients/Sub*
dtype0*-
_class#
!loc:@lstm_1/while/clip_by_value*
valueB
 *    *
_output_shapes
: 
�
/gradients/lstm_1/while/clip_by_value_grad/zerosFill1gradients/lstm_1/while/clip_by_value_grad/Shape_25gradients/lstm_1/while/clip_by_value_grad/zeros/Const*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_accStack*

stack_name *V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnterRefEnter<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
@gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush	StackPush?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter"lstm_1/while/clip_by_value/Minimum^gradients/Add*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Hgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop/RefEnterRefEnter<gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopStackPopHgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/clip_by_value'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1Stack*

stack_name *F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
�
Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1RefEnter>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
is_constant(
�
Bgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPush_1	StackPushAgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1lstm_1/while/Const^gradients/Add*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
T0*
swap_memory(*
_output_shapes
:
�
Jgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1/RefEnterRefEnter>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
is_constant(
�
Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1StackPopJgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1/RefEnter^gradients/Sub*F
_class<
:loc:@lstm_1/while/Constloc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
: 
�
6gradients/lstm_1/while/clip_by_value_grad/GreaterEqualGreaterEqual?gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPopAgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/StackPop_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
Egradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_accStack*

stack_name *-
_class#
!loc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
�
Hgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
is_constant(
�
Igradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPush	StackPushHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnter/gradients/lstm_1/while/clip_by_value_grad/Shape^gradients/Add*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
swap_memory(*
_output_shapes
:
�
Qgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/clip_by_value*
is_constant(
�
Hgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPopStackPopQgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*-
_class#
!loc:@lstm_1/while/clip_by_value*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgsBroadcastGradientArgsHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop1gradients/lstm_1/while/clip_by_value_grad/Shape_1*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*2
_output_shapes 
:���������:���������
�
0gradients/lstm_1/while/clip_by_value_grad/SelectSelect6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
4gradients/lstm_1/while/clip_by_value_grad/LogicalNot
LogicalNot6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:���������
�
2gradients/lstm_1/while/clip_by_value_grad/Select_1Select4gradients/lstm_1/while/clip_by_value_grad/LogicalNot)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/clip_by_value_grad/SumSum0gradients/lstm_1/while/clip_by_value_grad/Select?gradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs*-
_class#
!loc:@lstm_1/while/clip_by_value*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
1gradients/lstm_1/while/clip_by_value_grad/ReshapeReshape-gradients/lstm_1/while/clip_by_value_grad/SumHgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/StackPop*
Tshape0*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������
�
/gradients/lstm_1/while/clip_by_value_grad/Sum_1Sum2gradients/lstm_1/while/clip_by_value_grad/Select_1Agradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs:1*-
_class#
!loc:@lstm_1/while/clip_by_value*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
3gradients/lstm_1/while/clip_by_value_grad/Reshape_1Reshape/gradients/lstm_1/while/clip_by_value_grad/Sum_11gradients/lstm_1/while/clip_by_value_grad/Shape_1*
Tshape0*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*
_output_shapes
: 
�
)gradients/lstm_1/while/Tanh_grad/TanhGradTanhGrad.gradients/lstm_1/while/mul_6_grad/mul/StackPop+gradients/lstm_1/while/mul_6_grad/Reshape_1*$
_class
loc:@lstm_1/while/Tanh*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_6_grad/ShapeShapelstm_1/while/strided_slice_3*%
_class
loc:@lstm_1/while/add_6*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/add_6_grad/Shape_1Shapelstm_1/while/MatMul_3*%
_class
loc:@lstm_1/while/add_6*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
�
Agradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_6_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_6*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
�
@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
�
Cgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_6_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_6*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_6*
is_constant(
�
Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_6*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_6_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_6*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_6_grad/SumSum+gradients/lstm_1/while/mul_8_grad/Reshape_17gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_6*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/add_6_grad/ReshapeReshape%gradients/lstm_1/while/add_6_grad/Sum@gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_6*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_6_grad/Sum_1Sum+gradients/lstm_1/while/mul_8_grad/Reshape_19gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_6*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/add_6_grad/Reshape_1Reshape'gradients/lstm_1/while/add_6_grad/Sum_1Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/add_6*
T0*'
_output_shapes
:���������
�
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ShapeShapelstm_1/while/add_3*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0*
T0*
_output_shapes
:
�
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB *
_output_shapes
: 
�
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2Shape3gradients/lstm_1/while/clip_by_value_1_grad/Reshape*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
out_type0*
T0*
_output_shapes
:
�
?gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
valueB
 *    *
_output_shapes
: 
�
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zerosFill;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_2?gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros/Const*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
Cgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_accStack*

stack_name *P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Fgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Ggradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush	StackPushFgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_3^gradients/Add*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Ogradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Fgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopStackPopOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/add_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*'
_output_shapes
:���������
�
Egradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1RefEnterEgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPush_1	StackPushHgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_3^gradients/Add*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Qgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterEgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1StackPopQgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*R
_classH
Floc:@lstm_1/while/Const_3)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
: 
�
=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual	LessEqualFgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPopHgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/StackPop_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
Ogradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Rgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Sgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnter9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape^gradients/Add*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
[gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterOgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
is_constant(
�
Rgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPopStackPop[gradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
	elem_type0*
_output_shapes
:
�
Igradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*2
_output_shapes 
:���������:���������
�
:gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectSelect=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:���������
�
<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
7gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumSum:gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SelectIgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/ReshapeReshape7gradients/lstm_1/while/clip_by_value_1/Minimum_grad/SumRgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/StackPop*
Tshape0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������
�
9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1Sum<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Kgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs:1*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape_1Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Sum_1;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Shape_1*
Tshape0*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*
_output_shapes
: 
�
4gradients/lstm_1/while/Switch_3_grad_1/NextIterationNextIteration+gradients/lstm_1/while/mul_4_grad/Reshape_1*'
_class
loc:@lstm_1/while/Merge_3*
T0*'
_output_shapes
:���������
�
7gradients/lstm_1/while/clip_by_value/Minimum_grad/ShapeShapelstm_1/while/add_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
T0*
_output_shapes
:
�
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1Const^gradients/Sub*
dtype0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB *
_output_shapes
: 
�
9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2Shape1gradients/lstm_1/while/clip_by_value_grad/Reshape*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/ConstConst^gradients/Sub*
dtype0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
valueB
 *    *
_output_shapes
: 
�
7gradients/lstm_1/while/clip_by_value/Minimum_grad/zerosFill9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_2=gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros/Const*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
Agradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_accStack*

stack_name *N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnterRefEnterAgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Egradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush	StackPushDgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnterlstm_1/while/add_1^gradients/Add*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Mgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop/RefEnterRefEnterAgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Dgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopStackPopMgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop/RefEnter^gradients/Sub*N
_classD
Bloc:@lstm_1/while/add_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*'
_output_shapes
:���������
�
Cgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1Stack*

stack_name *P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1RefEnterCgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPush_1	StackPushFgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1lstm_1/while/Const_1^gradients/Add*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Ogradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1/RefEnterRefEnterCgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1StackPopOgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1/RefEnter^gradients/Sub*P
_classF
Dloc:@lstm_1/while/Const_1'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
: 
�
;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual	LessEqualDgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPopFgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/StackPop_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
Mgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_accStack*

stack_name *5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
Pgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnterRefEnterMgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Qgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPush	StackPushPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnter7gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape^gradients/Add*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
swap_memory(*
_output_shapes
:
�
Ygradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnterMgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
is_constant(
�
Pgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPopStackPopYgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgsBroadcastGradientArgsPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop9gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*2
_output_shapes 
:���������:���������
�
8gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectSelect;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot
LogicalNot;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:���������
�
:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Select<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
5gradients/lstm_1/while/clip_by_value/Minimum_grad/SumSum8gradients/lstm_1/while/clip_by_value/Minimum_grad/SelectGgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
9gradients/lstm_1/while/clip_by_value/Minimum_grad/ReshapeReshape5gradients/lstm_1/while/clip_by_value/Minimum_grad/SumPgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/StackPop*
Tshape0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������
�
7gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_1Sum:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Igradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs:1*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
;gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape_1Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/Sum_19gradients/lstm_1/while/clip_by_value/Minimum_grad/Shape_1*
Tshape0*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*
_output_shapes
: 
�
'gradients/lstm_1/while/add_4_grad/ShapeShapelstm_1/while/strided_slice_2*%
_class
loc:@lstm_1/while/add_4*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/add_4_grad/Shape_1Shapelstm_1/while/MatMul_2*%
_class
loc:@lstm_1/while/add_4*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
�
Agradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_4_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_4*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
�
@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
�
Cgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_4_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_4*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_4*
is_constant(
�
Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_4*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_4_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_4*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_4_grad/SumSum)gradients/lstm_1/while/Tanh_grad/TanhGrad7gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/add_4_grad/ReshapeReshape%gradients/lstm_1/while/add_4_grad/Sum@gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_4*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_4_grad/Sum_1Sum)gradients/lstm_1/while/Tanh_grad/TanhGrad9gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_4*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/add_4_grad/Reshape_1Reshape'gradients/lstm_1/while/add_4_grad/Sum_1Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/add_4*
T0*'
_output_shapes
:���������
�
1gradients/lstm_1/while/strided_slice_3_grad/ShapeShapelstm_1/while/TensorArrayReadV3*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
out_type0*
T0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_3*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_3_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
is_constant(
�
Egradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_3/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_3'loc:@lstm_1/while/strided_slice_3/stack*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_3/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_1*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_3/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_3)loc:@lstm_1/while/strided_slice_3/stack_2*
	elem_type0*
_output_shapes
:
�
<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_6_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_3
�
1gradients/lstm_1/while/MatMul_3_grad/MatMul/EnterEnterlstm_1/strided_slice_7*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_3*
is_constant(
�
+gradients/lstm_1/while/MatMul_3_grad/MatMulMatMul+gradients/lstm_1/while/add_6_grad/Reshape_11gradients/lstm_1/while/MatMul_3_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_3*
T0*'
_output_shapes
:���������
�
3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
�
6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
is_constant(
�
7gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnterlstm_1/while/mul_7^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
�
?gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
is_constant(
�
6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_3loc:@lstm_1/while/mul_7*
	elem_type0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_6_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_3*
T0*
_output_shapes

:
�
'gradients/lstm_1/while/add_3_grad/ShapeShapelstm_1/while/mul_3*%
_class
loc:@lstm_1/while/add_3*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/add_3_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_3*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_3*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_3*
is_constant(
�
Agradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_3_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_3*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_3*
is_constant(
�
@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_3*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_3_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_3_grad/Shape_1*%
_class
loc:@lstm_1/while/add_3*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_3_grad/SumSum;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape7gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/add_3_grad/ReshapeReshape%gradients/lstm_1/while/add_3_grad/Sum@gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_3*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_3_grad/Sum_1Sum;gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Reshape9gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/add_3_grad/Reshape_1Reshape'gradients/lstm_1/while/add_3_grad/Sum_1)gradients/lstm_1/while/add_3_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/add_3*
T0*
_output_shapes
: 
�
'gradients/lstm_1/while/add_1_grad/ShapeShapelstm_1/while/mul_1*%
_class
loc:@lstm_1/while/add_1*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/add_1_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/add_1*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_1*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_1*
is_constant(
�
Agradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_1_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_1*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_1*
is_constant(
�
@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_1*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_1_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/add_1_grad/Shape_1*%
_class
loc:@lstm_1/while/add_1*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_1_grad/SumSum9gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape7gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/add_1_grad/ReshapeReshape%gradients/lstm_1/while/add_1_grad/Sum@gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_1*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_1_grad/Sum_1Sum9gradients/lstm_1/while/clip_by_value/Minimum_grad/Reshape9gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/add_1_grad/Reshape_1Reshape'gradients/lstm_1/while/add_1_grad/Sum_1)gradients/lstm_1/while/add_1_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/add_1*
T0*
_output_shapes
: 
�
1gradients/lstm_1/while/strided_slice_2_grad/ShapeShapelstm_1/while/TensorArrayReadV3*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
out_type0*
T0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_2*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_2_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
is_constant(
�
Egradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_2*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_2/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_2'loc:@lstm_1/while/strided_slice_2/stack*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_2/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_1*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_2/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_2)loc:@lstm_1/while/strided_slice_2/stack_2*
	elem_type0*
_output_shapes
:
�
<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_4_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_2
�
1gradients/lstm_1/while/MatMul_2_grad/MatMul/EnterEnterlstm_1/strided_slice_6*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_2*
is_constant(
�
+gradients/lstm_1/while/MatMul_2_grad/MatMulMatMul+gradients/lstm_1/while/add_4_grad/Reshape_11gradients/lstm_1/while/MatMul_2_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_2*
T0*'
_output_shapes
:���������
�
3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
�
6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
is_constant(
�
7gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnterlstm_1/while/mul_5^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
T0*
swap_memory(*
_output_shapes
:
�
?gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
is_constant(
�
6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_2loc:@lstm_1/while/mul_5*
	elem_type0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/MatMul_2_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_4_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_2*
T0*
_output_shapes

:
�
'gradients/lstm_1/while/mul_7_grad/ShapeShapelstm_1/while/Identity_2*%
_class
loc:@lstm_1/while/mul_7*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/mul_7_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_7*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_7*
is_constant(
�
Agradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_7_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_7*
is_constant(
�
@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_7_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_7*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_7_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_7_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_7_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
is_constant(
�
/gradients/lstm_1/while/mul_7_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_7_grad/mul/RefEnterlstm_1/while/mul_7/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_7_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_7_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
is_constant(
�
.gradients/lstm_1/while/mul_7_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_7_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_7loc:@lstm_1/while/mul_7/y*
	elem_type0*
_output_shapes
: 
�
%gradients/lstm_1/while/mul_7_grad/mulMul+gradients/lstm_1/while/MatMul_3_grad/MatMul.gradients/lstm_1/while/mul_7_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_7_grad/SumSum%gradients/lstm_1/while/mul_7_grad/mul7gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_7_grad/ReshapeReshape%gradients/lstm_1/while/mul_7_grad/Sum@gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/mul_7_grad/mul_1/f_accStack*

stack_name *C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_7_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_7_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
is_constant(
�
1gradients/lstm_1/while/mul_7_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_7_grad/mul_1/RefEnterlstm_1/while/Identity_2^gradients/Add*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_7_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_7_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
is_constant(
�
0gradients/lstm_1/while/mul_7_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_7_grad/mul_1/StackPop/RefEnter^gradients/Sub*C
_class9
7loc:@lstm_1/while/Identity_2loc:@lstm_1/while/mul_7*
	elem_type0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_7_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_3_grad/MatMul*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_7_grad/Sum_1Sum'gradients/lstm_1/while/mul_7_grad/mul_19gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_7*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_7_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_7_grad/Sum_1)gradients/lstm_1/while/mul_7_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_7*
T0*
_output_shapes
: 
�
0gradients/lstm_1/while/MatMul_3/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
valueB*    *
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
is_constant( 
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_3/Enter_grad/NextIteration*
N*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0* 
_output_shapes
:: 
�
1gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*(
_output_shapes
::
�
.gradients/lstm_1/while/MatMul_3/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_3_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
�
8gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_3/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:
�
'gradients/lstm_1/while/mul_3_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_3*
valueB *
_output_shapes
: 
�
)gradients/lstm_1/while/mul_3_grad/Shape_1Shapelstm_1/while/add_2*%
_class
loc:@lstm_1/while/mul_3*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_3*
is_constant(
�
Agradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_3_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_3*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_3*
is_constant(
�
@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_3_grad/Shape@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_3_grad/mul/f_accStack*

stack_name *>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_3_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_3_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
is_constant(
�
/gradients/lstm_1/while/mul_3_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_3_grad/mul/RefEnterlstm_1/while/add_2^gradients/Add*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_3_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_3_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
is_constant(
�
.gradients/lstm_1/while/mul_3_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_3_grad/mul/StackPop/RefEnter^gradients/Sub*>
_class4
2loc:@lstm_1/while/add_2loc:@lstm_1/while/mul_3*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_3_grad/mulMul)gradients/lstm_1/while/add_3_grad/Reshape.gradients/lstm_1/while/mul_3_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_3_grad/SumSum%gradients/lstm_1/while/mul_3_grad/mul7gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_3_grad/ReshapeReshape%gradients/lstm_1/while/mul_3_grad/Sum'gradients/lstm_1/while/mul_3_grad/Shape*
Tshape0*%
_class
loc:@lstm_1/while/mul_3*
T0*
_output_shapes
: 
�
-gradients/lstm_1/while/mul_3_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_3_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_3_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
is_constant(
�
1gradients/lstm_1/while/mul_3_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_3_grad/mul_1/RefEnterlstm_1/while/mul_3/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_3_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_3_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
is_constant(
�
0gradients/lstm_1/while/mul_3_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_3_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_3loc:@lstm_1/while/mul_3/x*
	elem_type0*
_output_shapes
: 
�
'gradients/lstm_1/while/mul_3_grad/mul_1Mul0gradients/lstm_1/while/mul_3_grad/mul_1/StackPop)gradients/lstm_1/while/add_3_grad/Reshape*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_3_grad/Sum_1Sum'gradients/lstm_1/while/mul_3_grad/mul_19gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_3*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_3_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_3_grad/Sum_1@gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_1_grad/ShapeConst^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_1*
valueB *
_output_shapes
: 
�
)gradients/lstm_1/while/mul_1_grad/Shape_1Shapelstm_1/while/add*%
_class
loc:@lstm_1/while/mul_1*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_1*
is_constant(
�
Agradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnter)gradients/lstm_1/while/mul_1_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/mul_1*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_1*
is_constant(
�
@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs'gradients/lstm_1/while/mul_1_grad/Shape@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_1_grad/mul/f_accStack*

stack_name *<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_1_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_1_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
is_constant(
�
/gradients/lstm_1/while/mul_1_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_1_grad/mul/RefEnterlstm_1/while/add^gradients/Add*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_1_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_1_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
is_constant(
�
.gradients/lstm_1/while/mul_1_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_1_grad/mul/StackPop/RefEnter^gradients/Sub*<
_class2
0loc:@lstm_1/while/addloc:@lstm_1/while/mul_1*
	elem_type0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_1_grad/mulMul)gradients/lstm_1/while/add_1_grad/Reshape.gradients/lstm_1/while/mul_1_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_1_grad/SumSum%gradients/lstm_1/while/mul_1_grad/mul7gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_1_grad/ReshapeReshape%gradients/lstm_1/while/mul_1_grad/Sum'gradients/lstm_1/while/mul_1_grad/Shape*
Tshape0*%
_class
loc:@lstm_1/while/mul_1*
T0*
_output_shapes
: 
�
-gradients/lstm_1/while/mul_1_grad/mul_1/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
	elem_type0*
_output_shapes
:
�
0gradients/lstm_1/while/mul_1_grad/mul_1/RefEnterRefEnter-gradients/lstm_1/while/mul_1_grad/mul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
is_constant(
�
1gradients/lstm_1/while/mul_1_grad/mul_1/StackPush	StackPush0gradients/lstm_1/while/mul_1_grad/mul_1/RefEnterlstm_1/while/mul_1/x^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
T0*
swap_memory(*
_output_shapes
:
�
9gradients/lstm_1/while/mul_1_grad/mul_1/StackPop/RefEnterRefEnter-gradients/lstm_1/while/mul_1_grad/mul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
is_constant(
�
0gradients/lstm_1/while/mul_1_grad/mul_1/StackPopStackPop9gradients/lstm_1/while/mul_1_grad/mul_1/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_1loc:@lstm_1/while/mul_1/x*
	elem_type0*
_output_shapes
: 
�
'gradients/lstm_1/while/mul_1_grad/mul_1Mul0gradients/lstm_1/while/mul_1_grad/mul_1/StackPop)gradients/lstm_1/while/add_1_grad/Reshape*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_1_grad/Sum_1Sum'gradients/lstm_1/while/mul_1_grad/mul_19gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_1_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_1_grad/Sum_1@gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_5_grad/ShapeShapelstm_1/while/Identity_2*%
_class
loc:@lstm_1/while/mul_5*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/mul_5_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_5*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_5*
is_constant(
�
Agradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_5_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_5*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_5*
is_constant(
�
@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_5*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_5_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_5*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_5_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_5_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_5_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
is_constant(
�
/gradients/lstm_1/while/mul_5_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_5_grad/mul/RefEnterlstm_1/while/mul_5/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_5_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_5_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
is_constant(
�
.gradients/lstm_1/while/mul_5_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_5_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_5loc:@lstm_1/while/mul_5/y*
	elem_type0*
_output_shapes
: 
�
%gradients/lstm_1/while/mul_5_grad/mulMul+gradients/lstm_1/while/MatMul_2_grad/MatMul.gradients/lstm_1/while/mul_5_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_5_grad/SumSum%gradients/lstm_1/while/mul_5_grad/mul7gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_5_grad/ReshapeReshape%gradients/lstm_1/while/mul_5_grad/Sum@gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_5_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_2_grad/MatMul*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_5_grad/Sum_1Sum'gradients/lstm_1/while/mul_5_grad/mul_19gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_5*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_5_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_5_grad/Sum_1)gradients/lstm_1/while/mul_5_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_5*
T0*
_output_shapes
: 
�
0gradients/lstm_1/while/MatMul_2/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
valueB*    *
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
is_constant( 
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_2/Enter_grad/NextIteration*
N*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0* 
_output_shapes
:: 
�
1gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*(
_output_shapes
::
�
.gradients/lstm_1/while/MatMul_2/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_2_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
�
8gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_2/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:
�
+gradients/lstm_1/strided_slice_7_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_7*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_22gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_7
�
'gradients/lstm_1/while/add_2_grad/ShapeShapelstm_1/while/strided_slice_1*%
_class
loc:@lstm_1/while/add_2*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/add_2_grad/Shape_1Shapelstm_1/while/MatMul_1*%
_class
loc:@lstm_1/while/add_2*
out_type0*
T0*
_output_shapes
:
�
=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
�
Agradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/add_2_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/add_2*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
�
@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
�
?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1RefEnter?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
�
Cgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPush_1	StackPushBgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1)gradients/lstm_1/while/add_2_grad/Shape_1^gradients/Add*%
_class
loc:@lstm_1/while/add_2*
T0*
swap_memory(*
_output_shapes
:
�
Kgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/add_2*
is_constant(
�
Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1StackPopKgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/add_2*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/add_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPopBgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1*%
_class
loc:@lstm_1/while/add_2*
T0*2
_output_shapes 
:���������:���������
�
%gradients/lstm_1/while/add_2_grad/SumSum+gradients/lstm_1/while/mul_3_grad/Reshape_17gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/add_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/add_2_grad/ReshapeReshape%gradients/lstm_1/while/add_2_grad/Sum@gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/add_2*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/add_2_grad/Sum_1Sum+gradients/lstm_1/while/mul_3_grad/Reshape_19gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/add_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/add_2_grad/Reshape_1Reshape'gradients/lstm_1/while/add_2_grad/Sum_1Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*%
_class
loc:@lstm_1/while/add_2*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/add_grad/ShapeShapelstm_1/while/strided_slice*#
_class
loc:@lstm_1/while/add*
out_type0*
T0*
_output_shapes
:
�
'gradients/lstm_1/while/add_grad/Shape_1Shapelstm_1/while/MatMul*#
_class
loc:@lstm_1/while/add*
out_type0*
T0*
_output_shapes
:
�
;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_accStack*

stack_name *#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
�
>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnterRefEnter;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
�
?gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush	StackPush>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter%gradients/lstm_1/while/add_grad/Shape^gradients/Add*#
_class
loc:@lstm_1/while/add*
T0*
swap_memory(*
_output_shapes
:
�
Ggradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter;gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
�
>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPopStackPopGgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
�
=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1Stack*

stack_name *#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1RefEnter=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
�
Agradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPush_1	StackPush@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1'gradients/lstm_1/while/add_grad/Shape_1^gradients/Add*#
_class
loc:@lstm_1/while/add*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1/RefEnterRefEnter=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/add*
is_constant(
�
@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1StackPopIgradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/add*
	elem_type0*
_output_shapes
:
�
5gradients/lstm_1/while/add_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1*#
_class
loc:@lstm_1/while/add*
T0*2
_output_shapes 
:���������:���������
�
#gradients/lstm_1/while/add_grad/SumSum+gradients/lstm_1/while/mul_1_grad/Reshape_15gradients/lstm_1/while/add_grad/BroadcastGradientArgs*#
_class
loc:@lstm_1/while/add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
'gradients/lstm_1/while/add_grad/ReshapeReshape#gradients/lstm_1/while/add_grad/Sum>gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop*
Tshape0*#
_class
loc:@lstm_1/while/add*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/add_grad/Sum_1Sum+gradients/lstm_1/while/mul_1_grad/Reshape_17gradients/lstm_1/while/add_grad/BroadcastGradientArgs:1*#
_class
loc:@lstm_1/while/add*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/add_grad/Reshape_1Reshape%gradients/lstm_1/while/add_grad/Sum_1@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/StackPop_1*
Tshape0*#
_class
loc:@lstm_1/while/add*
T0*'
_output_shapes
:���������
�
+gradients/lstm_1/strided_slice_6_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_6*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_22gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_6
�
1gradients/lstm_1/while/strided_slice_1_grad/ShapeShapelstm_1/while/TensorArrayReadV3*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
out_type0*
T0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_accStack*

stack_name */
_class%
#!loc:@lstm_1/while/strided_slice_1*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush	StackPushEgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter1gradients/lstm_1/while/strided_slice_1_grad/Shape^gradients/Add*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
is_constant(
�
Egradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopStackPopNgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*/
_class%
#!loc:@lstm_1/while/strided_slice_1*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_1	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1"lstm_1/while/strided_slice_1/stack^gradients/Add*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*X
_classN
L!loc:@lstm_1/while/strided_slice_1'loc:@lstm_1/while/strided_slice_1/stack*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_2	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2$lstm_1/while/strided_slice_1/stack_1^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_1*
	elem_type0*
_output_shapes
:
�
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
	elem_type0*
_output_shapes
:
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3RefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
is_constant(
�
Hgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPush_3	StackPushGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3$lstm_1/while/strided_slice_1/stack_2^gradients/Add*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
T0*
swap_memory(*
_output_shapes
:
�
Pgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterDgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
is_constant(
�
Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3StackPopPgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*Z
_classP
N!loc:@lstm_1/while/strided_slice_1)loc:@lstm_1/while/strided_slice_1/stack_2*
	elem_type0*
_output_shapes
:
�
<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGradStridedSliceGradEgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPopGgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_1Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_2Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/StackPop_3)gradients/lstm_1/while/add_2_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask */
_class%
#!loc:@lstm_1/while/strided_slice_1
�
1gradients/lstm_1/while/MatMul_1_grad/MatMul/EnterEnterlstm_1/strided_slice_5*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*(
_class
loc:@lstm_1/while/MatMul_1*
is_constant(
�
+gradients/lstm_1/while/MatMul_1_grad/MatMulMatMul+gradients/lstm_1/while/add_2_grad/Reshape_11gradients/lstm_1/while/MatMul_1_grad/MatMul/Enter*
transpose_b(*
transpose_a( *(
_class
loc:@lstm_1/while/MatMul_1*
T0*'
_output_shapes
:���������
�
3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_accStack*

stack_name *A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
�
6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnterRefEnter3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
is_constant(
�
7gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPush	StackPush6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnterlstm_1/while/mul_2^gradients/Add*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
T0*
swap_memory(*
_output_shapes
:
�
?gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop/RefEnterRefEnter3gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
is_constant(
�
6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPopStackPop?gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*A
_class7
5loc:@lstm_1/while/MatMul_1loc:@lstm_1/while/mul_2*
	elem_type0*'
_output_shapes
:���������
�
-gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_2_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_1*
T0*
_output_shapes

:
�
/gradients/lstm_1/while/strided_slice_grad/ShapeShapelstm_1/while/TensorArrayReadV3*-
_class#
!loc:@lstm_1/while/strided_slice*
out_type0*
T0*
_output_shapes
:
�
@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_accStack*

stack_name *-
_class#
!loc:@lstm_1/while/strided_slice*
	elem_type0*
_output_shapes
:
�
Cgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnterRefEnter@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/strided_slice*
is_constant(
�
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush	StackPushCgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter/gradients/lstm_1/while/strided_slice_grad/Shape^gradients/Add*-
_class#
!loc:@lstm_1/while/strided_slice*
T0*
swap_memory(*
_output_shapes
:
�
Lgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop/RefEnterRefEnter@gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*-
_class#
!loc:@lstm_1/while/strided_slice*
is_constant(
�
Cgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopStackPopLgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop/RefEnter^gradients/Sub*-
_class#
!loc:@lstm_1/while/strided_slice*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1Stack*

stack_name *T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_1	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1 lstm_1/while/strided_slice/stack^gradients/Add*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
is_constant(
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1/RefEnter^gradients/Sub*T
_classJ
Hloc:@lstm_1/while/strided_slice%loc:@lstm_1/while/strided_slice/stack*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2Stack*

stack_name *V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_2	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2"lstm_1/while/strided_slice/stack_1^gradients/Add*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
is_constant(
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_1*
	elem_type0*
_output_shapes
:
�
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3Stack*

stack_name *V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
	elem_type0*
_output_shapes
:
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3RefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
is_constant(
�
Fgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPush_3	StackPushEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3"lstm_1/while/strided_slice/stack_2^gradients/Add*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
T0*
swap_memory(*
_output_shapes
:
�
Ngradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3/RefEnterRefEnterBgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
is_constant(
�
Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3StackPopNgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3/RefEnter^gradients/Sub*V
_classL
Jloc:@lstm_1/while/strided_slice'loc:@lstm_1/while/strided_slice/stack_2*
	elem_type0*
_output_shapes
:
�
:gradients/lstm_1/while/strided_slice_grad/StridedSliceGradStridedSliceGradCgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPopEgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_1Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_2Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/StackPop_3'gradients/lstm_1/while/add_grad/Reshape*
new_axis_mask *
Index0*'
_output_shapes
:���������`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *-
_class#
!loc:@lstm_1/while/strided_slice
�
/gradients/lstm_1/while/MatMul_grad/MatMul/EnterEnterlstm_1/strided_slice_4*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/MatMul*
is_constant(
�
)gradients/lstm_1/while/MatMul_grad/MatMulMatMul)gradients/lstm_1/while/add_grad/Reshape_1/gradients/lstm_1/while/MatMul_grad/MatMul/Enter*
transpose_b(*
transpose_a( *&
_class
loc:@lstm_1/while/MatMul*
T0*'
_output_shapes
:���������
�
1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_accStack*

stack_name *=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
�
4gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnterRefEnter1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
is_constant(
�
5gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPush	StackPush4gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnterlstm_1/while/mul^gradients/Add*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
T0*
swap_memory(*
_output_shapes
:
�
=gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop/RefEnterRefEnter1gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
is_constant(
�
4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPopStackPop=gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop/RefEnter^gradients/Sub*=
_class3
1loc:@lstm_1/while/MatMulloc:@lstm_1/while/mul*
	elem_type0*'
_output_shapes
:���������
�
+gradients/lstm_1/while/MatMul_grad/MatMul_1MatMul4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop)gradients/lstm_1/while/add_grad/Reshape_1*
transpose_b( *
transpose_a(*&
_class
loc:@lstm_1/while/MatMul*
T0*
_output_shapes

:
�
'gradients/lstm_1/while/mul_2_grad/ShapeShapelstm_1/while/Identity_2*%
_class
loc:@lstm_1/while/mul_2*
out_type0*
T0*
_output_shapes
:
�
)gradients/lstm_1/while/mul_2_grad/Shape_1Const^gradients/Sub*
dtype0*%
_class
loc:@lstm_1/while/mul_2*
valueB *
_output_shapes
: 
�
=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_accStack*

stack_name *%
_class
loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
�
@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnterRefEnter=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_2*
is_constant(
�
Agradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPush	StackPush@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnter'gradients/lstm_1/while/mul_2_grad/Shape^gradients/Add*%
_class
loc:@lstm_1/while/mul_2*
T0*
swap_memory(*
_output_shapes
:
�
Igradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter=gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*%
_class
loc:@lstm_1/while/mul_2*
is_constant(
�
@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPopStackPopIgradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*%
_class
loc:@lstm_1/while/mul_2*
	elem_type0*
_output_shapes
:
�
7gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgs@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop)gradients/lstm_1/while/mul_2_grad/Shape_1*%
_class
loc:@lstm_1/while/mul_2*
T0*2
_output_shapes 
:���������:���������
�
+gradients/lstm_1/while/mul_2_grad/mul/f_accStack*

stack_name *@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
	elem_type0*
_output_shapes
:
�
.gradients/lstm_1/while/mul_2_grad/mul/RefEnterRefEnter+gradients/lstm_1/while/mul_2_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
is_constant(
�
/gradients/lstm_1/while/mul_2_grad/mul/StackPush	StackPush.gradients/lstm_1/while/mul_2_grad/mul/RefEnterlstm_1/while/mul_2/y^gradients/Add*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
T0*
swap_memory(*
_output_shapes
:
�
7gradients/lstm_1/while/mul_2_grad/mul/StackPop/RefEnterRefEnter+gradients/lstm_1/while/mul_2_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
is_constant(
�
.gradients/lstm_1/while/mul_2_grad/mul/StackPopStackPop7gradients/lstm_1/while/mul_2_grad/mul/StackPop/RefEnter^gradients/Sub*@
_class6
4loc:@lstm_1/while/mul_2loc:@lstm_1/while/mul_2/y*
	elem_type0*
_output_shapes
: 
�
%gradients/lstm_1/while/mul_2_grad/mulMul+gradients/lstm_1/while/MatMul_1_grad/MatMul.gradients/lstm_1/while/mul_2_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_2_grad/SumSum%gradients/lstm_1/while/mul_2_grad/mul7gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs*%
_class
loc:@lstm_1/while/mul_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_2_grad/ReshapeReshape%gradients/lstm_1/while/mul_2_grad/Sum@gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/StackPop*
Tshape0*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_2_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_1_grad/MatMul*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/while/mul_2_grad/Sum_1Sum'gradients/lstm_1/while/mul_2_grad/mul_19gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs:1*%
_class
loc:@lstm_1/while/mul_2*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
+gradients/lstm_1/while/mul_2_grad/Reshape_1Reshape'gradients/lstm_1/while/mul_2_grad/Sum_1)gradients/lstm_1/while/mul_2_grad/Shape_1*
Tshape0*%
_class
loc:@lstm_1/while/mul_2*
T0*
_output_shapes
: 
�
0gradients/lstm_1/while/MatMul_1/Enter_grad/b_accConst*
dtype0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
valueB*    *
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
is_constant( 
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2Merge2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_18gradients/lstm_1/while/MatMul_1/Enter_grad/NextIteration*
N*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0* 
_output_shapes
:: 
�
1gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*(
_output_shapes
::
�
.gradients/lstm_1/while/MatMul_1/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_1_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
�
8gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_1/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:
�
gradients/AddN_3AddN<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad:gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad*
N*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*'
_output_shapes
:���������`
�
Ugradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterEnterlstm_1/TensorArray_1*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
�
Wgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1EnterAlstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant(
�
Ogradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3Ugradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/EnterWgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1^gradients/Sub*
source	gradients*'
_class
loc:@lstm_1/TensorArray_1*
_output_shapes

::
�
Kgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flowIdentityWgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3/Enter_1P^gradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
Qgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3TensorArrayWriteV3Ogradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/TensorArrayGradV3Zgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/StackPopgradients/AddN_3Kgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayGrad/gradient_flow*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
%gradients/lstm_1/while/mul_grad/ShapeShapelstm_1/while/Identity_2*#
_class
loc:@lstm_1/while/mul*
out_type0*
T0*
_output_shapes
:
�
'gradients/lstm_1/while/mul_grad/Shape_1Const^gradients/Sub*
dtype0*#
_class
loc:@lstm_1/while/mul*
valueB *
_output_shapes
: 
�
;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_accStack*

stack_name *#
_class
loc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
�
>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnterRefEnter;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/mul*
is_constant(
�
?gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPush	StackPush>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnter%gradients/lstm_1/while/mul_grad/Shape^gradients/Add*#
_class
loc:@lstm_1/while/mul*
T0*
swap_memory(*
_output_shapes
:
�
Ggradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop/RefEnterRefEnter;gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*#
_class
loc:@lstm_1/while/mul*
is_constant(
�
>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPopStackPopGgradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop/RefEnter^gradients/Sub*#
_class
loc:@lstm_1/while/mul*
	elem_type0*
_output_shapes
:
�
5gradients/lstm_1/while/mul_grad/BroadcastGradientArgsBroadcastGradientArgs>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop'gradients/lstm_1/while/mul_grad/Shape_1*#
_class
loc:@lstm_1/while/mul*
T0*2
_output_shapes 
:���������:���������
�
)gradients/lstm_1/while/mul_grad/mul/f_accStack*

stack_name *<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
	elem_type0*
_output_shapes
:
�
,gradients/lstm_1/while/mul_grad/mul/RefEnterRefEnter)gradients/lstm_1/while/mul_grad/mul/f_acc*
_output_shapes
:**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
is_constant(
�
-gradients/lstm_1/while/mul_grad/mul/StackPush	StackPush,gradients/lstm_1/while/mul_grad/mul/RefEnterlstm_1/while/mul/y^gradients/Add*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
T0*
swap_memory(*
_output_shapes
:
�
5gradients/lstm_1/while/mul_grad/mul/StackPop/RefEnterRefEnter)gradients/lstm_1/while/mul_grad/mul/f_acc*
_output_shapes
:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
is_constant(
�
,gradients/lstm_1/while/mul_grad/mul/StackPopStackPop5gradients/lstm_1/while/mul_grad/mul/StackPop/RefEnter^gradients/Sub*<
_class2
0loc:@lstm_1/while/mulloc:@lstm_1/while/mul/y*
	elem_type0*
_output_shapes
: 
�
#gradients/lstm_1/while/mul_grad/mulMul)gradients/lstm_1/while/MatMul_grad/MatMul,gradients/lstm_1/while/mul_grad/mul/StackPop*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:���������
�
#gradients/lstm_1/while/mul_grad/SumSum#gradients/lstm_1/while/mul_grad/mul5gradients/lstm_1/while/mul_grad/BroadcastGradientArgs*#
_class
loc:@lstm_1/while/mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
'gradients/lstm_1/while/mul_grad/ReshapeReshape#gradients/lstm_1/while/mul_grad/Sum>gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/StackPop*
Tshape0*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop)gradients/lstm_1/while/MatMul_grad/MatMul*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/while/mul_grad/Sum_1Sum%gradients/lstm_1/while/mul_grad/mul_17gradients/lstm_1/while/mul_grad/BroadcastGradientArgs:1*#
_class
loc:@lstm_1/while/mul*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
�
)gradients/lstm_1/while/mul_grad/Reshape_1Reshape%gradients/lstm_1/while/mul_grad/Sum_1'gradients/lstm_1/while/mul_grad/Shape_1*
Tshape0*#
_class
loc:@lstm_1/while/mul*
T0*
_output_shapes
: 
�
.gradients/lstm_1/while/MatMul/Enter_grad/b_accConst*
dtype0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
valueB*    *
_output_shapes

:
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter.gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
_output_shapes

:*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
is_constant( 
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2Merge0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_16gradients/lstm_1/while/MatMul/Enter_grad/NextIteration*
N*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0* 
_output_shapes
:: 
�
/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*(
_output_shapes
::
�
,gradients/lstm_1/while/MatMul/Enter_grad/AddAdd1gradients/lstm_1/while/MatMul/Enter_grad/Switch:1+gradients/lstm_1/while/MatMul_grad/MatMul_1*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
�
6gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration,gradients/lstm_1/while/MatMul/Enter_grad/Add*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit/gradients/lstm_1/while/MatMul/Enter_grad/Switch*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:
�
+gradients/lstm_1/strided_slice_5_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_5*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_22gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_5
�
;gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_accConst*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*
valueB
 *    *
_output_shapes
: 
�
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Enter;gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc*
_output_shapes
: *4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*'
_class
loc:@lstm_1/TensorArray_1*
is_constant( 
�
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2Merge=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_1Cgradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIteration*
N*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: : 
�
<gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/SwitchSwitch=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_2gradients/b_count_2*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: : 
�
9gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/AddAdd>gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch:1Qgradients/lstm_1/while/TensorArrayReadV3_grad/TensorArrayWrite/TensorArrayWriteV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
Cgradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/NextIterationNextIteration9gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Add*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3Exit<gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/Switch*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
gradients/AddN_4AddN)gradients/lstm_1/while/mul_7_grad/Reshape)gradients/lstm_1/while/mul_5_grad/Reshape)gradients/lstm_1/while/mul_2_grad/Reshape'gradients/lstm_1/while/mul_grad/Reshape*
N*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������
�
+gradients/lstm_1/strided_slice_4_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_4*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_20gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_4
�
rgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3TensorArrayGradV3lstm_1/TensorArray_1=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3*
source	gradients*'
_class
loc:@lstm_1/TensorArray_1*
_output_shapes

::
�
ngradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flowIdentity=gradients/lstm_1/while/TensorArrayReadV3/Enter_1_grad/b_acc_3s^gradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3*'
_class
loc:@lstm_1/TensorArray_1*
T0*
_output_shapes
: 
�
dgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV3TensorArrayGatherV3rgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/TensorArrayGradV3lstm_1/TensorArrayUnstack/rangengradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGrad/gradient_flow*
element_shape:*
dtype0*'
_class
loc:@lstm_1/TensorArray_1*+
_output_shapes
:0���������`
�
4gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_4*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:���������
�
gradients/AddN_5AddN6gradients/lstm_1/strided_slice_7_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_6_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_5_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*
N*)
_class
loc:@lstm_1/strided_slice_7*
T0*
_output_shapes

:`
�
1gradients/lstm_1/transpose_grad/InvertPermutationInvertPermutationlstm_1/transpose/perm*#
_class
loc:@lstm_1/transpose*
T0*
_output_shapes
:
�
)gradients/lstm_1/transpose_grad/transpose	Transposedgradients/lstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3_grad/TensorArrayGatherV31gradients/lstm_1/transpose_grad/InvertPermutation*#
_class
loc:@lstm_1/transpose*
Tperm0*
T0*+
_output_shapes
:���������0`
�
!gradients/lstm_1/concat_grad/RankConst*
dtype0* 
_class
loc:@lstm_1/concat*
value	B :*
_output_shapes
: 
�
 gradients/lstm_1/concat_grad/modFloorModlstm_1/concat/axis!gradients/lstm_1/concat_grad/Rank* 
_class
loc:@lstm_1/concat*
T0*
_output_shapes
: 
�
"gradients/lstm_1/concat_grad/ShapeShapelstm_1/Reshape_1* 
_class
loc:@lstm_1/concat*
out_type0*
T0*
_output_shapes
:
�
#gradients/lstm_1/concat_grad/ShapeNShapeNlstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7*
N*
out_type0* 
_class
loc:@lstm_1/concat*,
_output_shapes
::::*
T0
�
)gradients/lstm_1/concat_grad/ConcatOffsetConcatOffset gradients/lstm_1/concat_grad/mod#gradients/lstm_1/concat_grad/ShapeN%gradients/lstm_1/concat_grad/ShapeN:1%gradients/lstm_1/concat_grad/ShapeN:2%gradients/lstm_1/concat_grad/ShapeN:3*
N* 
_class
loc:@lstm_1/concat*,
_output_shapes
::::
�
"gradients/lstm_1/concat_grad/SliceSlice)gradients/lstm_1/transpose_grad/transpose)gradients/lstm_1/concat_grad/ConcatOffset#gradients/lstm_1/concat_grad/ShapeN*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������0
�
$gradients/lstm_1/concat_grad/Slice_1Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:1%gradients/lstm_1/concat_grad/ShapeN:1*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������0
�
$gradients/lstm_1/concat_grad/Slice_2Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:2%gradients/lstm_1/concat_grad/ShapeN:2*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������0
�
$gradients/lstm_1/concat_grad/Slice_3Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:3%gradients/lstm_1/concat_grad/ShapeN:3*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������0
�
%gradients/lstm_1/Reshape_1_grad/ShapeShapelstm_1/BiasAdd*#
_class
loc:@lstm_1/Reshape_1*
out_type0*
T0*
_output_shapes
:
�
'gradients/lstm_1/Reshape_1_grad/ReshapeReshape"gradients/lstm_1/concat_grad/Slice%gradients/lstm_1/Reshape_1_grad/Shape*
Tshape0*#
_class
loc:@lstm_1/Reshape_1*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/Reshape_3_grad/ShapeShapelstm_1/BiasAdd_1*#
_class
loc:@lstm_1/Reshape_3*
out_type0*
T0*
_output_shapes
:
�
'gradients/lstm_1/Reshape_3_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_1%gradients/lstm_1/Reshape_3_grad/Shape*
Tshape0*#
_class
loc:@lstm_1/Reshape_3*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/Reshape_5_grad/ShapeShapelstm_1/BiasAdd_2*#
_class
loc:@lstm_1/Reshape_5*
out_type0*
T0*
_output_shapes
:
�
'gradients/lstm_1/Reshape_5_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_2%gradients/lstm_1/Reshape_5_grad/Shape*
Tshape0*#
_class
loc:@lstm_1/Reshape_5*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/Reshape_7_grad/ShapeShapelstm_1/BiasAdd_3*#
_class
loc:@lstm_1/Reshape_7*
out_type0*
T0*
_output_shapes
:
�
'gradients/lstm_1/Reshape_7_grad/ReshapeReshape$gradients/lstm_1/concat_grad/Slice_3%gradients/lstm_1/Reshape_7_grad/Shape*
Tshape0*#
_class
loc:@lstm_1/Reshape_7*
T0*'
_output_shapes
:���������
�
)gradients/lstm_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_1_grad/Reshape*!
_class
loc:@lstm_1/BiasAdd*
data_formatNHWC*
T0*
_output_shapes
:
�
+gradients/lstm_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_3_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_1*
data_formatNHWC*
T0*
_output_shapes
:
�
+gradients/lstm_1/BiasAdd_2_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_5_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_2*
data_formatNHWC*
T0*
_output_shapes
:
�
+gradients/lstm_1/BiasAdd_3_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_7_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_3*
data_formatNHWC*
T0*
_output_shapes
:
�
#gradients/lstm_1/MatMul_grad/MatMulMatMul'gradients/lstm_1/Reshape_1_grad/Reshapelstm_1/strided_slice*
transpose_b(*
transpose_a( * 
_class
loc:@lstm_1/MatMul*
T0*'
_output_shapes
:���������
�
%gradients/lstm_1/MatMul_grad/MatMul_1MatMullstm_1/Reshape'gradients/lstm_1/Reshape_1_grad/Reshape*
transpose_b( *
transpose_a(* 
_class
loc:@lstm_1/MatMul*
T0*
_output_shapes

:
�
+gradients/lstm_1/strided_slice_8_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_8*
valueB:`*
_output_shapes
:
�
6gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2)gradients/lstm_1/BiasAdd_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:`*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_8
�
%gradients/lstm_1/MatMul_1_grad/MatMulMatMul'gradients/lstm_1/Reshape_3_grad/Reshapelstm_1/strided_slice_1*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_1*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/MatMul_1_grad/MatMul_1MatMullstm_1/Reshape_2'gradients/lstm_1/Reshape_3_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_1*
T0*
_output_shapes

:
�
+gradients/lstm_1/strided_slice_9_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_9*
valueB:`*
_output_shapes
:
�
6gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2+gradients/lstm_1/BiasAdd_1_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:`*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_9
�
%gradients/lstm_1/MatMul_2_grad/MatMulMatMul'gradients/lstm_1/Reshape_5_grad/Reshapelstm_1/strided_slice_2*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_2*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/MatMul_2_grad/MatMul_1MatMullstm_1/Reshape_4'gradients/lstm_1/Reshape_5_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_2*
T0*
_output_shapes

:
�
,gradients/lstm_1/strided_slice_10_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_10*
valueB:`*
_output_shapes
:
�
7gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2+gradients/lstm_1/BiasAdd_2_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:`*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask **
_class 
loc:@lstm_1/strided_slice_10
�
%gradients/lstm_1/MatMul_3_grad/MatMulMatMul'gradients/lstm_1/Reshape_7_grad/Reshapelstm_1/strided_slice_3*
transpose_b(*
transpose_a( *"
_class
loc:@lstm_1/MatMul_3*
T0*'
_output_shapes
:���������
�
'gradients/lstm_1/MatMul_3_grad/MatMul_1MatMullstm_1/Reshape_6'gradients/lstm_1/Reshape_7_grad/Reshape*
transpose_b( *
transpose_a(*"
_class
loc:@lstm_1/MatMul_3*
T0*
_output_shapes

:
�
,gradients/lstm_1/strided_slice_11_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_11*
valueB:`*
_output_shapes
:
�
7gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2+gradients/lstm_1/BiasAdd_3_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes
:`*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask **
_class 
loc:@lstm_1/strided_slice_11
�
)gradients/lstm_1/strided_slice_grad/ShapeConst*
dtype0*'
_class
loc:@lstm_1/strided_slice*
valueB"   `   *
_output_shapes
:
�
4gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad)gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2%gradients/lstm_1/MatMul_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *'
_class
loc:@lstm_1/strided_slice
�
+gradients/lstm_1/strided_slice_1_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_1*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2'gradients/lstm_1/MatMul_1_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_1
�
+gradients/lstm_1/strided_slice_2_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_2*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2'gradients/lstm_1/MatMul_2_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_2
�
+gradients/lstm_1/strided_slice_3_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_3*
valueB"   `   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2'gradients/lstm_1/MatMul_3_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes

:`*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask *)
_class
loc:@lstm_1/strided_slice_3
�
gradients/AddN_6AddN6gradients/lstm_1/strided_slice_8_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_9_grad/StridedSliceGrad7gradients/lstm_1/strided_slice_10_grad/StridedSliceGrad7gradients/lstm_1/strided_slice_11_grad/StridedSliceGrad*
N*)
_class
loc:@lstm_1/strided_slice_8*
T0*
_output_shapes
:`
�
gradients/AddN_7AddN"gradients/lstm_1/Square_grad/mul_14gradients/lstm_1/strided_slice_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_1_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_2_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_3_grad/StridedSliceGrad*
N* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes

:`
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
�
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
valueB*    *
_output_shapes

:
~

Variable_1
VariableV2*
dtype0*
shape
:*
shared_name *
	container *
_output_shapes

:
�
Variable_1/AssignAssign
Variable_1Const_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes

:
T
Const_4Const*
dtype0*
valueB`*    *
_output_shapes
:`
v

Variable_2
VariableV2*
dtype0*
shape:`*
shared_name *
	container *
_output_shapes
:`
�
Variable_2/AssignAssign
Variable_2Const_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:`
k
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes
:`
\
Const_5Const*
dtype0*
valueB`*    *
_output_shapes

:`
~

Variable_3
VariableV2*
dtype0*
shape
:`*
shared_name *
	container *
_output_shapes

:`
�
Variable_3/AssignAssign
Variable_3Const_5*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes

:`
o
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes

:`
\
Const_6Const*
dtype0*
valueB`*    *
_output_shapes

:`
~

Variable_4
VariableV2*
dtype0*
shape
:`*
shared_name *
	container *
_output_shapes

:`
�
Variable_4/AssignAssign
Variable_4Const_6*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:`
o
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes

:`
J
mul_2Mulrho/readVariable/read*
T0*
_output_shapes
:
L
sub_1/xConst*
dtype0*
valueB
 *  �?*
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
�
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
 *  �*
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
 *w�+2*
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
�
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

:
L
sub_3/xConst*
dtype0*
valueB
 *  �?*
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

:
F
mul_6Mulsub_3Square_2*
T0*
_output_shapes

:
C
add_4Addmul_5mul_6*
T0*
_output_shapes

:
�
Assign_2Assign
Variable_1add_4*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:
f
mul_7Mullr/read&gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:
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
 *  �*
_output_shapes
: 
\
clip_by_value_1/MinimumMinimumadd_4Const_10*
T0*
_output_shapes

:
e
clip_by_value_1Maximumclip_by_value_1/MinimumConst_9*
T0*
_output_shapes

:
H
Sqrt_1Sqrtclip_by_value_1*
T0*
_output_shapes

:
L
add_5/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
F
add_5AddSqrt_1add_5/y*
T0*
_output_shapes

:
G
div_2RealDivmul_7add_5*
T0*
_output_shapes

:
Q
sub_4Subdense_1/kernel/readdiv_2*
T0*
_output_shapes

:
�
Assign_3Assigndense_1/kernelsub_4*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:
L
mul_8Mulrho/readVariable_2/read*
T0*
_output_shapes
:`
L
sub_5/xConst*
dtype0*
valueB
 *  �?*
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
:`
B
mul_9Mulsub_5Square_3*
T0*
_output_shapes
:`
?
add_6Addmul_8mul_9*
T0*
_output_shapes
:`
�
Assign_4Assign
Variable_2add_6*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes
:`
M
mul_10Mullr/readgradients/AddN_6*
T0*
_output_shapes
:`
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
 *  �*
_output_shapes
: 
X
clip_by_value_2/MinimumMinimumadd_6Const_12*
T0*
_output_shapes
:`
b
clip_by_value_2Maximumclip_by_value_2/MinimumConst_11*
T0*
_output_shapes
:`
D
Sqrt_2Sqrtclip_by_value_2*
T0*
_output_shapes
:`
L
add_7/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
B
add_7AddSqrt_2add_7/y*
T0*
_output_shapes
:`
D
div_3RealDivmul_10add_7*
T0*
_output_shapes
:`
J
sub_6Sublstm_1/bias/readdiv_3*
T0*
_output_shapes
:`
�
Assign_5Assignlstm_1/biassub_6*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes
:`
Q
mul_11Mulrho/readVariable_3/read*
T0*
_output_shapes

:`
L
sub_7/xConst*
dtype0*
valueB
 *  �?*
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

:`
G
mul_12Mulsub_7Square_4*
T0*
_output_shapes

:`
E
add_8Addmul_11mul_12*
T0*
_output_shapes

:`
�
Assign_6Assign
Variable_3add_8*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes

:`
Q
mul_13Mullr/readgradients/AddN_7*
T0*
_output_shapes

:`
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
 *  �*
_output_shapes
: 
\
clip_by_value_3/MinimumMinimumadd_8Const_14*
T0*
_output_shapes

:`
f
clip_by_value_3Maximumclip_by_value_3/MinimumConst_13*
T0*
_output_shapes

:`
H
Sqrt_3Sqrtclip_by_value_3*
T0*
_output_shapes

:`
L
add_9/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
F
add_9AddSqrt_3add_9/y*
T0*
_output_shapes

:`
H
div_4RealDivmul_13add_9*
T0*
_output_shapes

:`
P
sub_8Sublstm_1/kernel/readdiv_4*
T0*
_output_shapes

:`
�
Assign_7Assignlstm_1/kernelsub_8*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes

:`
Q
mul_14Mulrho/readVariable_4/read*
T0*
_output_shapes

:`
L
sub_9/xConst*
dtype0*
valueB
 *  �?*
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

:`
G
mul_15Mulsub_9Square_5*
T0*
_output_shapes

:`
F
add_10Addmul_14mul_15*
T0*
_output_shapes

:`
�
Assign_8Assign
Variable_4add_10*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes

:`
Q
mul_16Mullr/readgradients/AddN_5*
T0*
_output_shapes

:`
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
 *  �*
_output_shapes
: 
]
clip_by_value_4/MinimumMinimumadd_10Const_16*
T0*
_output_shapes

:`
f
clip_by_value_4Maximumclip_by_value_4/MinimumConst_15*
T0*
_output_shapes

:`
H
Sqrt_4Sqrtclip_by_value_4*
T0*
_output_shapes

:`
M
add_11/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
H
add_11AddSqrt_4add_11/y*
T0*
_output_shapes

:`
I
div_5RealDivmul_16add_11*
T0*
_output_shapes

:`
[
sub_10Sublstm_1/recurrent_kernel/readdiv_5*
T0*
_output_shapes

:`
�
Assign_9Assignlstm_1/recurrent_kernelsub_10*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes

:`
�
group_deps_1NoOp^add_1^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
�
initNoOp^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^lstm_1/bias/Assign^dense_1/kernel/Assign^dense_1/bias/Assign
^lr/Assign^rho/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign""�
	variables��
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
Variable_4:0Variable_4/AssignVariable_4/read:0"˃
while_context����
��
lstm_1/while/lstm_1/while/  *lstm_1/while/LoopCond:02lstm_1/while/Merge:0:lstm_1/while/Identity:0Blstm_1/while/Exit:0Blstm_1/while/Exit_1:0Blstm_1/while/Exit_2:0Blstm_1/while/Exit_3:0Bgradients/f_count_2:0Jс
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
lstm_1/while/strided_slice_3:0�
Ggradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc_1:0Jgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter_1:0a
-gradients/lstm_1/while/mul_7_grad/mul/f_acc:00gradients/lstm_1/while/mul_7_grad/mul/RefEnter:0e
/gradients/lstm_1/while/mul_3_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_3_grad/mul_1/RefEnter:0�
Egradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/f_acc:0Hgradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual/RefEnter:0�
Egradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc_1:0Hgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter_1:0�
Ggradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc_1:0Jgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter_1:0�
?gradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_7_grad/BroadcastGradientArgs/RefEnter:0�
?gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc_1:0Bgradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter_1:0�
?gradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter:0�
?gradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter:0�
?gradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_5_grad/BroadcastGradientArgs/RefEnter:0�
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_3:0Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_3:0�
?gradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_1_grad/BroadcastGradientArgs/RefEnter:0q
5gradients/lstm_1/while/MatMul_3_grad/MatMul_1/f_acc:08gradients/lstm_1/while/MatMul_3_grad/MatMul_1/RefEnter:0�
Agradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter_1:0a
-gradients/lstm_1/while/mul_4_grad/mul/f_acc:00gradients/lstm_1/while/mul_4_grad/mul/RefEnter:09
lstm_1/strided_slice_5:0lstm_1/while/MatMul_1/Enter:0�
@gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc:0Cgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter:0q
5gradients/lstm_1/while/MatMul_1_grad/MatMul_1/f_acc:08gradients/lstm_1/while/MatMul_1_grad/MatMul_1/RefEnter:0�
=gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/f_acc:0@gradients/lstm_1/while/mul_grad/BroadcastGradientArgs/RefEnter:0�
@gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc_1:0Cgradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter_1:0�
?gradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_2_grad/BroadcastGradientArgs/RefEnter:0�
Bgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc:0Egradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter:0�
Agradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter_1:0�
?gradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_1_grad/BroadcastGradientArgs/RefEnter:09
lstm_1/strided_slice_6:0lstm_1/while/MatMul_2/Enter:0�
Egradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/f_acc:0Hgradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual/RefEnter:0�
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_2:0Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_2:0�
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_2:0Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_2:0e
/gradients/lstm_1/while/mul_4_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_4_grad/mul_1/RefEnter:0�
Bgradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/f_acc_1:0Egradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual/RefEnter_1:0�
Igradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/f_acc:0Lgradients/lstm_1/while/clip_by_value_2_grad/BroadcastGradientArgs/RefEnter:0�
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_2:0Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_2:0�
Agradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter_1:0q
5gradients/lstm_1/while/MatMul_2_grad/MatMul_1/f_acc:08gradients/lstm_1/while/MatMul_2_grad/MatMul_1/RefEnter:0�
Dgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc:0Ggradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter:0a
-gradients/lstm_1/while/mul_2_grad/mul/f_acc:00gradients/lstm_1/while/mul_2_grad/mul/RefEnter:0�
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_1:0Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_1:0�
Igradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/f_acc:0Lgradients/lstm_1/while/clip_by_value_1_grad/BroadcastGradientArgs/RefEnter:0a
-gradients/lstm_1/while/mul_9_grad/mul/f_acc:00gradients/lstm_1/while/mul_9_grad/mul/RefEnter:0�
>gradients/lstm_1/while/clip_by_value_grad/GreaterEqual/f_acc:0Agradients/lstm_1/while/clip_by_value_grad/GreaterEqual/RefEnter:0�
Qgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/f_acc:0Tgradients/lstm_1/while/clip_by_value_2/Minimum_grad/BroadcastGradientArgs/RefEnter:0�
?gradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_3_grad/BroadcastGradientArgs/RefEnter:0�
Dgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc:0Ggradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter:07
lstm_1/strided_slice_4:0lstm_1/while/MatMul/Enter:0@
lstm_1/TensorArray_1:0&lstm_1/while/TensorArrayReadV3/Enter:0�
?gradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_6_grad/BroadcastGradientArgs/RefEnter:0�
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_3:0Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_3:06
lstm_1/strided_slice_12:0lstm_1/while/Less/Enter:0e
/gradients/lstm_1/while/mul_8_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_8_grad/mul_1/RefEnter:0m
3gradients/lstm_1/while/MatMul_grad/MatMul_1/f_acc:06gradients/lstm_1/while/MatMul_grad/MatMul_1/RefEnter:0�
?gradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_5_grad/BroadcastGradientArgs/RefEnter:0�
Cgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/f_acc:0Fgradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual/RefEnter:0�
?gradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter:0a
-gradients/lstm_1/while/mul_1_grad/mul/f_acc:00gradients/lstm_1/while/mul_1_grad/mul/RefEnter:0e
/gradients/lstm_1/while/mul_6_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_6_grad/mul_1/RefEnter:0�
?gradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter:0P
lstm_1/TensorArray:08lstm_1/while/TensorArrayWrite/TensorArrayWriteV3/Enter:0�
Ogradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/f_acc:0Rgradients/lstm_1/while/clip_by_value/Minimum_grad/BroadcastGradientArgs/RefEnter:0e
/gradients/lstm_1/while/mul_9_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_9_grad/mul_1/RefEnter:0a
-gradients/lstm_1/while/mul_6_grad/mul/f_acc:00gradients/lstm_1/while/mul_6_grad/mul/RefEnter:0e
/gradients/lstm_1/while/mul_7_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_7_grad/mul_1/RefEnter:0�
Bgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc_1:0Egradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter_1:0a
-gradients/lstm_1/while/mul_3_grad/mul/f_acc:00gradients/lstm_1/while/mul_3_grad/mul/RefEnter:0�
=gradients/lstm_1/while/add_grad/BroadcastGradientArgs/f_acc:0@gradients/lstm_1/while/add_grad/BroadcastGradientArgs/RefEnter:0]
+gradients/lstm_1/while/mul_grad/mul/f_acc:0.gradients/lstm_1/while/mul_grad/mul/RefEnter:0a
-gradients/lstm_1/while/mul_8_grad/mul/f_acc:00gradients/lstm_1/while/mul_8_grad/mul/RefEnter:0�
Qgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/f_acc:0Tgradients/lstm_1/while/clip_by_value_1/Minimum_grad/BroadcastGradientArgs/RefEnter:0�
Dgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc:0Ggradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter:0�
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_3:0Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_3:0o
Clstm_1/TensorArrayUnstack/TensorArrayScatter/TensorArrayScatterV3:0(lstm_1/while/TensorArrayReadV3/Enter_1:0�
Agradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/mul_9_grad/BroadcastGradientArgs/RefEnter_1:09
lstm_1/strided_slice_7:0lstm_1/while/MatMul_3/Enter:0�
@gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/f_acc:0Cgradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual/RefEnter:0�
?gradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_7_grad/BroadcastGradientArgs/RefEnter:0�
Fgradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/f_acc_2:0Igradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad/RefEnter_2:0�
Fgradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/f_acc_1:0Igradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad/RefEnter_1:0�
?gradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_8_grad/BroadcastGradientArgs/RefEnter:0�
Agradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/add_4_grad/BroadcastGradientArgs/RefEnter_1:0�
Agradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/add_6_grad/BroadcastGradientArgs/RefEnter_1:0�
Fgradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/f_acc_1:0Igradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad/RefEnter_1:0�
?gradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/mul_3_grad/BroadcastGradientArgs/RefEnter:0�
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_3:0Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_3:0a
-gradients/lstm_1/while/mul_5_grad/mul/f_acc:00gradients/lstm_1/while/mul_5_grad/mul/RefEnter:0e
/gradients/lstm_1/while/mul_1_grad/mul_1/f_acc:02gradients/lstm_1/while/mul_1_grad/mul_1/RefEnter:0�
Ggradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/f_acc:0Jgradients/lstm_1/while/clip_by_value_grad/BroadcastGradientArgs/RefEnter:0�
Agradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/f_acc_1:0Dgradients/lstm_1/while/mul_4_grad/BroadcastGradientArgs/RefEnter_1:0�
?gradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/f_acc:0Bgradients/lstm_1/while/add_2_grad/BroadcastGradientArgs/RefEnter:0�
Dgradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/f_acc_1:0Ggradients/lstm_1/while/strided_slice_grad/StridedSliceGrad/RefEnter_1:0�
Ygradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/f_acc:0\gradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3/RefEnter:0"�
trainable_variables��
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
Variable_4:0Variable_4/AssignVariable_4/read:0ĺV       �K"	B+9G�A*

loss�59=F��       Oa�(	�D+9G�A*
	
lr
�#<��"       ���	G+9G�A*

val_loss�ߤ<M%�b       ��-	[�C9G�A*

loss�a�<��I(       �K"	��C9G�A*
	
lr
�#<��Y       ��2	t�C9G�A*

val_lossWy�<)T        ��-	7[	9G�A*

loss���<���       �K"	;9[	9G�A*
	
lr
�#<|~4j       ��2	C;[	9G�A*

val_loss��{<����       ��-	��r
9G�A*

loss2*�<R6A�       �K"	��r
9G�A*
	
lr
�#<
�9{       ��2	*s
9G�A*

val_loss~�<vR��       ��-	r9G�A*

loss+̑</tv       �K"	�ĉ9G�A*
	
lr
�#<��Z	       ��2	TƉ9G�A*

val_loss �P<�ɜ       ��-	���9G�A*

lossD<�<-�;s       �K"	Ū�9G�A*
	
lr
�#<��a(       ��2	��9G�A*

val_lossJo�<,�[       ��-	��9G�A*

loss��<D��       �K"		�9G�A*
	
lr
�#<�ie�       ��2	�9G�A*

val_lossG�<,s��       ��-	ʈ�9G�A*

lossаo<�T��       �K"	��9G�A*
	
lr
�#<?t�       ��2	���9G�A*

val_loss�ˠ<�hS�       ��-	��9G�A*

loss��<Dj�       �K"	R��9G�A*
	
lr
�#<y 3       ��2	���9G�A*

val_loss�P�<�2f�       ��-	Gs�9G�A	*

lossKq�<�-D1       �K"	�u�9G�A	*
	
lr
�#<	��       ��2	�w�9G�A	*

val_losszG\<1�V       ��-	�\9G�A
*

lossrfi<!��       �K"	�^9G�A
*
	
lr
�#<8_�       ��2	�`9G�A
*

val_loss��<��A       ��-	O,9G�A*

loss��n<#��X       �K"	&Q,9G�A*
	
lr
�#<��?       ��2	�R,9G�A*

val_loss�i<<�d�^       ��-	h?C9G�A*

lossd<Ta��       �K"	�AC9G�A*
	
lr
�#<^A 0       ��2	|CC9G�A*

val_loss�y=\�       ��-	�Z9G�A*

lossp�]<ys��       �K"	;�Z9G�A*
	
lr
�#<��e�       ��2	?�Z9G�A*

val_loss��7<�p}       ��-	(br9G�A*

loss[z`<�lϙ       �K"	Edr9G�A*
	
lr
�#<�9>!       ��2	4fr9G�A*

val_lossm: =F
J       ��-	
L�9G�A*

loss�c<8�-       �K"	jN�9G�A*
	
lr
�#<X
�+       ��2	<P�9G�A*

val_loss8>=a�C�       ��-	8/�9G�A*

loss��^<c�~1       �K"	r1�9G�A*
	
lr
�#<�W��       ��2	�3�9G�A*

val_lossl=��v       ��-	�F�9G�A*

loss��S<bqP       �K"	(I�9G�A*
	
lr
�#<j��o       ��2	ZK�9G�A*

val_lossp��<�h�K       ��-	4�9G�A*

loss%�N<z���       �K"	L6�9G�A*
	
lr
�#<g�       ��2	�8�9G�A*

val_lossDN�< 7${       ��-	��9G�A*

loss��^<�8+�       �K"	#��9G�A*
	
lr
�#<(��v       ��2	f��9G�A*

val_loss,�0<H���       ��-	Ts�9G�A*

loss��D<@4s       �K"	du�9G�A*
	
lr
�#<�;�       ��2	Ww�9G�A*

val_loss��=mD�[       ��-	�)9G�A*

lossz(X<�%��       �K"	�+9G�A*
	
lr
�#<ܾ��       ��2	�-9G�A*

val_loss�#i<��p�       ��-	�*9G�A*

loss��P<p�O       �K"	�*9G�A*
	
lr
�#<����       ��2	*9G�A*

val_lossZ�L<�Q       ��-	�A 9G�A*

loss�oQ<9��~       �K"	�A 9G�A*
	
lr
�#<��Lt       ��2	�A 9G�A*

val_lossYh�<��       ��-	d�W!9G�A*

loss&�H<��Μ       �K"	��W!9G�A*
	
lr
�#<Zm��       ��2	V�W!9G�A*

val_loss���<ȱ�       ��-	9o"9G�A*

loss K<��JZ       �K"	2;o"9G�A*
	
lr
�#<w�zm       ��2	=o"9G�A*

val_loss�<&<up�       ��-	k,�#9G�A*

loss�AL<`؁f       �K"	�.�#9G�A*
	
lr
�#<�\3       ��2	�0�#9G�A*

val_loss��(<3���       ��-	J�$9G�A*

loss�N<�� �       �K"	@L�$9G�A*
	
lr
�#<.��^       ��2	@N�$9G�A*

val_loss�m<���4       ��-	��%9G�A*

loss~LF<��gg       �K"	��%9G�A*
	
lr
�#<V!i�       ��2	�
�%9G�A*

val_lossk�;<���u       ��-	z�&9G�A*

loss*r<<��       �K"	��&9G�A*
	
lr
�#<��}9       ��2	��&9G�A*

val_loss.��<,CV�       ��-	K9�'9G�A*

loss�^O<ɇ�e       �K"	�;�'9G�A*
	
lr
�#<��dy       ��2	�=�'9G�A*

val_loss�{^<�NY�       ��-	P��(9G�A*

loss�E<��g       �K"	m��(9G�A*
	
lr
�#<�g�       ��2	6��(9G�A*

val_lossM9-<�ܣo       ��-	�*9G�A *

loss�><����       �K"	Q�*9G�A *
	
lr
�#<-73|       ��2	��*9G�A *

val_loss��<?q5r       ��-	r�+9G�A!*

loss�:C<�L֭       �K"	� +9G�A!*
	
lr
�#<����       ��2	O+9G�A!*

val_loss���;��U�       ��-	�G5,9G�A"*

loss!�=<�n��       �K"	J5,9G�A"*
	
lr
�#<bm"�       ��2	IL5,9G�A"*

val_loss`g<��b;       ��-	O�K-9G�A#*

lossg�G<S;Y       �K"	g�K-9G�A#*
	
lr
�#<$9r        ��2	$�K-9G�A#*

val_loss�C<p�y�       ��-	b.9G�A$*

lossq�?<�
�(       �K"	&b.9G�A$*
	
lr
�#<�\�a       ��2	�b.9G�A$*

val_lossia<5�Fq       ��-	�vx/9G�A%*

lossܹ><��M+       �K"	yx/9G�A%*
	
lr
�#<$�
�       ��2	F{x/9G�A%*

val_loss�R#<j��       ��-	�%�09G�A&*

loss�m><�'�       �K"	�'�09G�A&*
	
lr
�#<�W�       ��2	�)�09G�A&*

val_loss�X<_5�       ��-	��19G�A'*

loss�C<�7n       �K"	��19G�A'*
	
lr
�#<���       ��2	��19G�A'*

val_loss7�-<噹}       ��-	��29G�A(*

lossp�9<YP�       �K"	��29G�A(*
	
lr
�#<T�       ��2	��29G�A(*

val_loss:�<^��       ��-	�#�39G�A)*

loss�kF<,���       �K"	B&�39G�A)*
	
lr
�#<�5�       ��2	_(�39G�A)*

val_loss�� <��~       ��-	���49G�A**

loss_�=<���       �K"	���49G�A**
	
lr
�#<] o       ��2	���49G�A**

val_lossw�<1�e�       ��-	�!�59G�A+*

lossEQ7<-�       �K"	�#�59G�A+*
	
lr
�#<�u�       ��2	1&�59G�A+*

val_lossp��<�Fs�       ��-	f�79G�A,*

lossmB<3�-�       �K"	�79G�A,*
	
lr
�#<�b˂       ��2	��79G�A,*

val_lossW;g<2��       ��-	�l*89G�A-*

lossh�%<�x}d       �K"	�n*89G�A-*
	
lrn;׶d�       ��2	�p*89G�A-*

val_loss��-<$���       ��-	��?99G�A.*

lossE�$<nj�       �K"	�?99G�A.*
	
lrn;a� B       ��2	��?99G�A.*

val_losswQz<Ԫ��       ��-	�`V:9G�A/*

loss��$<p��C       �K"	�bV:9G�A/*
	
lrn;��U       ��2	=eV:9G�A/*

val_loss�5<��       ��-	Bo;9G�A0*

loss�#<�Jh       �K"	$Do;9G�A0*
	
lrn;�T��       ��2	�Eo;9G�A0*

val_lossO<-�]�       ��-	�v�<9G�A1*

loss.!<�f�5       �K"	�x�<9G�A1*
	
lrn;i���       ��2	�z�<9G�A1*

val_loss��Q<��c�