       �K"	  �6DI�Abrain.Event:2�d<	K     S�	�6DI�A"��
b
lstm_1_inputPlaceholder*
dtype0*
shape: *+
_output_shapes
:���������
l
lstm_1/random_uniform/shapeConst*
dtype0*
valueB"   �   *
_output_shapes
:
^
lstm_1/random_uniform/minConst*
dtype0*
valueB
 *�E�*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
dtype0*
valueB
 *�E>*
_output_shapes
: 
�
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
dtype0*
seed2���*
seed���)*
T0*
_output_shapes
:	�
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
T0*
_output_shapes
: 
�
lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
T0*
_output_shapes
:	�
|
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes
:	�
�
lstm_1/kernel
VariableV2*
dtype0*
shape:	�*
	container *
shared_name *
_output_shapes
:	�
�
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes
:	�
y
lstm_1/kernel/readIdentitylstm_1/kernel* 
_class
loc:@lstm_1/kernel*
T0*
_output_shapes
:	�
U
lstm_1/SquareSquarelstm_1/kernel/read*
T0*
_output_shapes
:	�
Q
lstm_1/mul/xConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
X

lstm_1/mulMullstm_1/mul/xlstm_1/Square*
T0*
_output_shapes
:	�
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
��
%lstm_1/recurrent_kernel/initial_valueConst*
dtype0*��
value��B��	(�"����_���D<���=��;���=]Go��޽��I��-O�qQ��.�<�s�:�E��� =�������c�;�r	���,��OݽD��2��<=6��<�K>����6�7�nۤ<ٵ뽂���|���T=���<�o���߼=��o���=f�	=��<V��<{#��<?��5�e�iߓ����P���r��3Q9�����~}� Tz�8�!ѳ=�\�=��۽���L�ӽI�=���=�E�=W\=ɇ�=ް4=`�?=S�ݼtY��*�=����M�<Ӛ/=u�=z ¼s9=��Ͻ���HT�ڶJ��=U��<Bdؽ�/>��нT��<`%����ec�<�ϣ;圡<	��<�Q��)2=�rI<ঁ�
5d���̽1[��xm�7�=�7�=mW�=�M;���#��R)�l�,�o>+�j|��_�����=32{=w\K����E�n=��=2�!���<y�A���>t�w��s��*>��;���w��=�(�iW��6׻��ҽ�N�=��>�~=�νX?���M���=Q�.>#D=f�=qSR��=�h�=�����J=>�^<���E`=.ݻ7�s=��c�����'Ծ��ι������
=d���*x�I\�=�=��m�Z�@_a���=��=�7��2>�=خ&�[�ٻ��c>��k��
�=?��;�ө��u�=�#Ὠ1��w~�������=#;�<bR��p<�����B�=ӳ,��c�<U��=}�a� 4|<��}�ؒ�=�u���=��K:ݖ=��Ľ�0���\�=)ܜ=?z1�A�<=�ΰ<g:�=���=��=>��=�1==;)<��9=�Ҵ����<L�\���=1B��I�-=�J�<߁�<r���>�+<n��>A�Ֆ����o��r0>lo���K�=���=T��=��4�l�z=Z/�����^�G��	iz=�vX<�
�O
=U���X�=�!�LD�=�\�<��9=������<��<=T*=(P뽼%�_�x��g��z^>-�*>G����Z�I>d=A%�=�K��R>�����.���皽RD���=;x��K'��^=���7��gg�=��Ec= �-=����R�(�(>n�<�n��9?����������C>��	>Y�d=@�<�1D=~��A�w��3<�d���T����*�=^�:��M���ț=3� =���<d��=�n��`����eN=�[�=�٧�I������.��*㼢� =69���)����<G��;E�Y=�:��D��o�=��ѽTQ	<1�!>�=��=�����;}(X��(H=tc`�Q�<�TR=�x�<����I�,��_˼v��G?�Y�=e���\w�<�Ԑ�D��=���_=�����=�k>�Up=��L���@�3�缵�N=�ݼsٹ�ui]>d�=:=�67;a��i�;�鼽RB�<?�O��
�X҇�b��=������q�;�$L�.^����½ױ =����3s�=����m�R<ž����7�Q�$>IXӽu�W�3-
>�� ���1��l<�g����;=E��X6<5
��=�\h�}0����=O��/�C=lJ�{���כ���������%�<G��b>U��=D(Ľ�I˻���=V^]�����d���J�=>�=�ʆ<�?�;�i�=\�X��\�<�
�^��=��=[3�=�1p=k�<�9>Pk]�!���"�>���<<N=�	��7�>��?�=�������#=�C=5Ғ�E�U=�c��ĽQE<$�R���=>͚>|������=$������Mt�n�+�:���w༛X�=7ʒ��=.�K�RG�:����d罭&���<wns=�s�=0�`;���=�y�<��=o�<l������
�=��Q/��W�@F>��'���U�F���l>�!�<c+U���-=��={���!��*0=<�=|+�< 7�=n�=,����z�=��=�;����^0=��7��ӕ�<)��<���6a�=.`;� >Ϥ5�`���θ�;��]<m�̼��I���=s{�ms>4�&��5�<��a>��o�Dz=�>\�ڽ�=��w=���y�y;R.�=�c=�(1�Fﴼ��m<�Wm<�y��Z�x�����,&=i��`ˬ<|�c�K��=<P�=�W�C ����=N�$���|��;q������=���<���=D���G�<��x<k��=���=;�н�{X�2�ۼ�B�9@9�;������B=N>�3�;�2g�J*�=!�=��;%B��̽�1ֽ���y|��k�=�8U<g�N<=���X=��y<�
s�<|��D<��.�o1��	<$@ս���=�-�2�X���;�/�Z;+��� �$������<��=/A�=w�=d�I=$���&����<"J>���=٫=D6�=伽R��<ܤ�;E�E<�H4=mM�m���΀=j�a�Ԍ��ȵ>���K���6��Rd���"���>�yB�I�[=�Ǥ��b�"�f=�8f;�C�=3J��@�;,{���h��>r�<s��<Dj1<�&޼������?=:F�:7R$=���w#��d��'��=��<�ȟ�N���qU	���=�I���]���6%��	>7�S����<��������9�ٽ	
漲9ҽU2��j��ؼ1�<Տ�=�5=�Ȋ;u<���D�o��Š<'Z=�uʽ	�.=����q��$�ս\�!��e,���ռ��%��U<��J��U�}76=��������m;�n>4��:^�����)Wջ��Q����<�t<!Z��(>���Ng"�0d�=o;=���=.���M����E����ܼ�=����%��}&��C网O�̙V�~�3��_���me=���;n ��u.�d@˽�NսsI'=�!0=��ƽ��]�,5>�6��R�a�=������>;R��=����E-=�2>�c�;v�U=��=��1����$�q�u= ���G'����p�<�.��:Ҏ�w>�-�:�߽������9>"L >�<� 9<Wnټ�(���L��\�=s���g\=���<A�<RH��V�a��Ƽk ���s�:rk����fO�g�G��ׅ���=��D=��<xk�/�;��Q����&>�c�<=Ce���g�Q>��=b��gG<��~���=Iw�=b!N��㠽�����C��2�<n| >�\>3l�<�N��]��=�#�=�#�==��,>�X�;=�.���t�R8սz)����V=	ݯ����[��V��4�	���>o %��c1�O�(=/@<D��ԗ���S�<p�����A= p��7\
>n�>r]��n���������)ԑ=��<%3K��u)��b <g��;��$9B����>��`��p���߆�e��U=�x+=�1Խ}HY=���<uB����=��=s��l+�A�ͼX6P���o��C�=��޽�M�;`�㼜�߼����(�|d=���z��=o��=���=a�=�{��4dȽU</�U=-��PY�=ނ=v��=�<=�L�;6�ƼC�e=�����z٪<�W���N�`�#�D�W=:��<�<�;5�+B=������=��;=��-�>�	���zO>Ecw��1ڽ1F�<:��|@:=JK��J@l��J��o�z=kb����>��<m��=��=ܚ�=��=mw�t_E�UӖ<\�u����Z4�p/�=r"�V��< v��������;	w�=��(=$�?>�&�=��`=�ƽ���=���<����SϽ�I	�K�<|B��Z��� �=V:>��~����_s=���=���QC�<Q��=�Z�=�uA���=p�ɽRL==މ
���|=��>BJF>�[=l��C=�>�M=7M/=�>�:����<W�=�=��)=Z����<Z����� �w\�
"$��.>\>�=L��k�=��1��
�1*�<H��=!��=[i��ǣ�=-�۽�g�=F�=<i.Ͻd��<T}
<����x�<�N=��z�hw�=�B�=�Zc��R=[��;ز���;<GK��y�=���W0h=�0M=í�f�ѼDP���;����Kч�,�=M>\�⼞���1P��A!��W�����/����">HM�=P'R���?= p���e�ㅢ=��=yKG=�$��פ��b>݅�=�hL=]���\�m�g�<�퉽�T�2x�t��=���<
̫��q���U�=@77>4�󽀼>�Ȃ=-�8��Z��<{��<\��Bz�;�9=�l��B>��A}8�~H��n������S�+I�=�7�<�W�
,�<Ӷ�;|�p������o~=����I�V�_6ȼ�Ҁ�ﯽ��_=�
u�斒��|�<^=3�>�>f$���n���Lh=P�ż�J�=o�[� ���^��-�ͼŊ!����n@G�J͔=N�X�&u.>���x��=��<e�#>Kr���=���Z�5=���=�M��Q�ȼIǧ;���L� ���>�K�5Ox���=c��=�̻�
�OV����=���=m�i<'��t�T=����1,�����O=�y��A��= &=��7�@B�;4�ֽ�k<���=������=����������fzV=[ن=��j���ݼ�=�м������=���=�y&����=.�;c�=K#	>C�!>�f˽Go>?��=ـ!��&�=p�����=�??�>]w�=��,�pOw>KA<�]��B���DJ���ƽ�M��Ž=��ƽa�����<���<��9<�W�;�_�S׻t���
:�]D:gYw=���;��z���>ʾP=�m�<Lμ��|��I5=t=h�����;�Yl���V<�A`��@}=�wI=�B��?�=�ɂ=Q���r<�{�<K)�=��3���'�uX�=�0=ݕ�����<V�u�	5��#�>��=��<�6Ƚ\�~=g����\��)�=��=<���=C�>{=�q����K�1$�=�n��-5�{^X>�&=ǜg<��>�����;οa;3����Of=���<S%0=�	f=v⥼�T�&͉=敹=⟏����=�(�=i;�H�����֦;a=�=���,�8�	v=���<��<i�	AS=�k#=7��!�<��X.�;�X<�=yA�=lv���3;������ѽT���a�=4�!<�����~m��m�[36�i~�̻��<q=����9C#�)�=���<吼�)�=g�>U񕽦�м��h=0C�<2Q�=Ǆ�=��=5��=��<�>�������;��<>���=w�;=AD��QQ��]'=����,� ��=z����Խ6�e=��<6�轉ϡ='*���m��
���
=���
a�=X�/=�0=J4-==Vd���=gJ�=�=�H��(�W�򮽰y�<�q"��D��J���@����<Կ��!.�<E��=�5���9�=1��=;�Q;ݰ���B��M��"�	>Yݒ=��m����<���D-�|Ý�o=�=�g=i U�u�Y=�N�<)PĽt�=wSۼ��=���=�`��J����1�T���;�5WC>�}۽��;;�:1�봽�R<HX��/a=��3=Q7(�������:��<�Ϳ�N^�bJ�=b�=p� =���;D=)�.��pJ���ڽ�b�:�<��=>l�g=BV�=��l=�88��%����X�h݉=�B=V3�<q���ռ��d�a8#=T �0Ӽ�Pҽ�I!��s=mS<=��=(��)���k*=��G��޹=��q��Mj��덽�m�'M=�]��6��=��;�����Z=�^w=\Ն=���b���=��	 �=�����%���n8�p*<�j=�����y;��a+>������8�P�=�⪽k8I=�yn���μ��_=� <+/<��Cֻ�̃��q���k>z,�����{⺼_s=�9�6�<{�!����<�s��`=R��=>_=�N�=2(>s�=O40=��)=<�\>�I>���s��=�?t<[�>R �<	�=��>��=�M=�B�ΗU< O�����dg�=x1Z<�;B<��=���<=e;����k��o����=��&>�\=�c=]-�:j��<Ku=�ܽy	��N0���׽ޓ��GG>^Tl=�)ؼ;a����;�.��״�0�?<���=�c�=�i�,��=��i��n�n��vo=?�=� =�#=�+=�L�<3ݣ����#�2=.�=0%Ӽ�;��
G<� >�?�=�n[=0=�-�̂�=�=�
�=�p�=�J�=B�=r�>���A���>�i�;B�=��=.�$��b��$�<Wx�xԟ�+G<d���ի���<�(���o=�o�=�=sõ�{�=� =�Ў�l�-�l��=��<�`�`�s<��>K����Y�򎼼W�<���=�� >o�;>�];���=���<�g��؄�=p	~="^/���=��.��g�~�����<�>Z�8�;��=��Ż0�<=��^�q�=����*����o��o�׻�:�=�U�=ێ�=3�:��]��=����2��o��=�kϼ~\q�B9�5���a��6�==\�=���Y�庈�D��&��'O�؁k<>�<���<�8>>��
���<�Q�þ�=��;�b<��]�B"��Hz=e�7=�2�=���<�]=u}=�}�=02=WR�=�Z���g��<9�.��R�����=.�
>&�$��L<�F@=���=��=P��<�e�=f�d����@�bE��d���3�=�=K2>�g�Ri=@=���=�쉻�~=*�Ľ�˯��t��H"V=@��=㤘=Em�=���<\)�=;ҽR��<��ѽX�o=��#���a<��˽��=4��=����oٗ���};{=Ž;�ǖ��`��x���P�<��j��Ȟ� ͔���W<�6����=�zν�a�=��=�����f+<����1�=0	�<0ż���="�#�U���ځD��,��H���G�=�ݼHm�L"�<d����0�<�#>���P�5=�1��o=�<,��� �a�A=z޺=Xu�������=�1x=�*=�={vѽ<BQ�o!�=����=XF�����4��o ]=��꽤*?=�>^���Dμ�o�Q3�tN�<	����O���B��)Q����X�=�@=ߠ�=X�x�C���SR�<�%<���<*d�=�<��>2�'�mܼ��=����)�b1�=p.R;Ksɽ��5��,+�� ��FDj=���5=T����O!�@�=s�X��Gd�t�=z=���=��=�(S>�< 輕�+=u�<Ƥ��@Ž��9o��Ґ��&>�]�=X$ڽ��ӽ "n<�D�<�-��У=(Žiq�=�׽�\>q4;\�#���]='e<sX���#�r�>�R������lt$=6���k�i�J�(�U�]��Z{<��K�/��<�i>�~��]�=�_�=��=�Ws�o��Pӳ��29=߫n<M�������(B�W�
����=|���[�-(9�w����+��E>N����=*w��ʤ�v��<��=p>�ɻ;���<��=(X;�ş��N��<㢾=�����=d�=��i�j[ֽ�$�=`��=П=��<�=�O����<�]��=�p<`��<\Y&=9z)<X;���9=
;�5�;?^&>��=�(=�֑=��<Jn����弒��=��=����[�5��=n1	>�����<$�4���9�0��<q�=�!�=z�=
5üP�;#6<�;k8����=��=A-=?==��<0"�������]�=�&=���N��= -'<bX���S1��B�=�;�\��<��߽HE<���=pEO��1���r>�*)>�mB=|�=h_1���>�QN��Aֽ��F�Y=ŵ������*����|T��s��rX=pC�<�>��=��>Q�=����������<�����4�=�y�=A�彇ķ��6���=���<��=�'ػ�# >�5�=�T>�9����ʽ:�=�c2���r�*ץ=q���D׽���sʡ=��$��"6��,�=�{�<�9��,��of ��Wv�=ü�x����Hu=�:�Kp�ht��p������I#�=o2z��O����#Z����q����/�="��<t�#=T�t<����@��ɢ�=ޛ�=�y����$>��P�[��O�g��E�=���%�Ľ��<���=�J��#�=�9n=x{=w����:>�4�<v|�4g���:�X��t�,>����*2�<�ɵ<Y�������{��- ��8��=��W��Z.=�#�<�G��)�!�����j�8�T<��!>��p�<���=��?��o�2轾ټ=��n=[E�=Z���&*��䴽�~�=N����=�@]<�*�mZ�=�K��4�=�wG����=[��=k0+���->Z¯�-���b��<p��=����:9��ꑽ�u˼L�%����� Y=O+��� �=$ ���?T=*��"�
����ٝ�<�'��\l�<��	���=8��� �=�~a<���<��A�1�ɼ	��v׼
�>���=ʦ<)��<�q�=_�"=>$���<�"�h�)<�Z4=A�o���.-�Ư_=�p=��=fSg="'���=�>��8�mY���l=��*�m{�=�>
���۽dh��,�d������>c�a=�Q=�����*��=�ڨ�?����=/r�<(+=<e�\n>]��=2��=���<�〽�v�= ���օC�Zڽ�\G�N�Z�<��(=��:�F�I�M�>n�ļaY=hY=B-�sm�=Ѧ���#�=i�j����= C>.`�+4_�b�
k�=BD��j*��R��Vev=y�E=u��<R����䛽v'�<��V=k�=��R=���;-�����:�%=>6�=���z=���=����y;��a�i�ȼn��=���=h_$<zQ�ra�<J��<��T=�l�<1�=&�(�/�>�,����M�T�'���G>�p�Ή�<��%=�ϫ=l= ���m4�=^��
d�;u�+>�G�;���=]v��U�=������<���=tI��}8�a�{�bj=�Z�=�K�=�hr�MqS��&>�Ht=����gdY�{{>�ST=�m�=��=})y��`J=����f�q�v������<�����5��)�=x�T�Y�
=��>H}E�^�t�O�
>�䣽���'�ӽÔ���]<�p���B>CU���]m���i�0����i�f������0^=7G=�2�<�����45�(D������=�7<��F��=!
<!���p�ߕɽkkм7m޽�=~!;FwW�v=��f�=a�r��<���G�ѽ��<�nG=�M>��]=O!���<
x���^���C>dP��#��</.v�[ȸ����=����:�<����g�Z�W='�ὡn7>�ٽbӼ���<?`�=-FF<��߼��=�|�腵=��o�/��=���&�=k����}=�?����Y��O������νۅ������m��<�Z2���\{�=�� ���=f��<�0�=Wr+=2�q=�A�8��=(�|�mI
>z^X=�1���U��د��߼4��<K�=�^���!�<�5���<E�={����F4��.��t�v:#Ib=�3�=�~�Β=���=y�Ҽ��J;��&;���=�rM=���=�[�:I�S��w�=Nf��,x��,=2W�Z ]=��h=�&��x!��Dh���Ȼ�	���H=���=񔂾:�>� �=a1i���4�]F�=]��=���9���7�.��%�?w =��=*��3���� ��F��a����B����w�j��A>#�V�����п=�y���ٽR�x=�&_�~���϶�:�2�=ȷ��称=�bt<O$?��k��Bq=����=d,�=E��=Å������.��fǁ���ʺaK���|/<��½�$�
�����+Ż�%����g�����&�,>��+=�,��K�B;�d"�sYz������#�=��!=�E`<s���i1��)0�"�Q<�A<r"
>�Ӽ$uP<���=���<+G�}p�=f.Q��><��<	�bb�;���f�;=ڋ��?�=Α#>޽>��=��=��:�0��;�����8���<�h>��=��N;a:��BQ�<���-���/�H;`{����=Zۅ={���4=�$���Ǻ�"/�z�s<@Jݽ �Ͻ��+O�=-���4�=EG=T	�:�;?t�l޶<V�=�4�QH�QrƽI9����E������>��=�<��I��A7�������4;�C �2½>�=�A�=cDB=��ʽ�i=��1=�&=�jX�2�3<�'�����=������< 3m�դ޽*��=��Ek�=���=�-��J�ui8=�=�t.>��D���{����=��Ž�(�<����(�<8	��p�*�k=�m�1K�b�>د}=@`�<U.:�Uz=ɨW�4������=����Ͻ���`e�M�%��䅼�' >Hx�<�U�(1��A��;�����o>��q�=հ6=$ʽ�Qz=����_=�e>'�=���`�<ڠ���������)�\���0?�����	>��p���=�=UC=yOX<.���"��n�m�욌=�����v��T=0�����:�|<�����-����p�������$�c1=��=�=�W=4��_��h��0�_�mۢ��)=10�=]��=�<�<~<�=���=���*�D�1�5��1���	ʽo��6��<�[=�Ff=W����?=D=os>���<Xq������=3P<�
�=�&���.>�!f=���=���<vh>R+>�"J;24���y;=(�I��<\/f<�[�b, �!i>�� =	�=H��a̶��Ű<��*��\�<a�Ӽ7��D��Y����>Kvٽ�v���=�,�=+���2����,���e�B�a��-�&�9��]�=�c��.�=wv�/�<腚=� �&�=��I=�)�=ͪ>!.%>RD?��+��OV��<��<�l����=Q !=(��XX�;fY=M�:_���k��c�=�j>>^<�m�j�td\��$�<m�^�F���j������n=��+���ż�	Ž��&�Y�.=+��= ��<?m��h���	V >ӛb�k�D<c8<<��k��ï;�"=����O+E=Zi=B�A=E�=�/e�HO�59�<g��Q9��w�;�@�=��C���=G�7���Ľ�
��`�Y�H�<�=˼���8�>f�н����J��US�_�=��<af	�,�߽=�ɉ9���=m%>�kҺK�U�޶T�zv=��<�ټ�a��] �Ϋ�=
@��FL�=6bk=$�=�,>+��|�g=s��;<�=��=tlǼϼol�=p�����<�$=Y;�i�@��=��=D��=�=<9=b�4�ò���=8iJ=�S�<4lɽ}}�����=3�;{���}g���K�<#��;K��;;�g����=\[@�����A�=�2o��E�;����U�=�t�D�o�^��έ�<i���ǎ�8���yr�=�m��y=�"X>�u�=�x�<wc�����;0�<��%��x�;���=�d���=�)�2}K�`?"��#һf����<$�߽O@=�����=�X�<�X��H!�0��?�L��&>!�<�<_=�޺<��<D��<:N�=1j�ʹ�=�	潽�f=v�>�P6�����tV��F�=_��5�=�0�=�;���o���4�=��<��U�����{�B���]=܍'>0� =pH\<� �=���hyӼ�N=�	�����=�K��,HT=;����
�N����tc��b<��2=�K�<*�.<{U3=TX�p�;Qy�=h�8F��=�<���=��罦J�V߽�@�<;Z�=�j>�s���+h���׺k�	>W_	>鑈=cħ<+�p<�s</.�=炆�L�i��O� ��=^6��ܤ:��t<��<�r=%{:��ۻ4�V��P,��;F�1=��j=R(����?���Y�b�b�<lˣ=������=߲,>~n��.Y�� ="�W���=-x#=J�=ߊ=��^��>��ƽ��{� ?���=��O�So㼄���������p�=��2��	=z�Ľ�6<b��I@��6e��7zŽ$$+��1��o{V=��ܽ��{=���;'�=�9=E�{��g$=��7<�mѽ���<�F�Ʃ����-;���= ��xƇ=��P=�=�p;=��<|L�=4@���S�=ϵS��"<���X�\=U������; ��=�-ҽƢ��m��=�"���*n=������=F�K�#��.�D=ԂX=pz/�p�����=��E�Q��N�=�罝�>3t�]��<)k�=s%�=��J=�!7�0�=��-��">�,�>N�:���=�QV=�ؼS�;�B��F<���Q=z	̽6�=�=�=
e>�j:�����n=�H�=�=(=��8�O=�=�=I)�<�=S�$����b=��ǽkI]>>��E�<����O�N�'
29g�7��%>w������Ζ=�}��h=�>dj�^9>,I&=P�=�I��ܟ�=�,�=���=@ҽ���+���ǜ=��=XV�9���ӷ�=o
v��mm���o=� [=��+>J(��?��<%�=̨=��==7�<1M���\�~�;��>F�	=I3\=�x=:���u��=����t6�=�M=1�l=�Ġ��8�=Z�N�s���MG=:c-��L4=�,;O,(=p��=c��=�e=��<��&<\���<_E����=�o��RP�=����T����=܃�;��]���P=f-�>����:>@�;;�����=�%�<�NS=,E=U?�=����W=m�=Q��=u��x��T�o$=��7m���~�$&{�����Y<}���Ck�=��N�μ��=��>z�<��C=���=����{���Mؽ�t�=������o
��dm�>�>r�=55w��m�=��1=�h/�BĊ=������=@���q���нg����<�8>�p�=4)�dX�����<&ݲ�(�=�{�Jg����<�Ϥ=�	>3�;�5�>=�����~�D��<6��;�չ��=���<�ؽ"�!=&�=0��=
߉�d?�<r�&=).>������=p�ҽW�C>�를�	"=fQ�=,�=l����=��=O�%<O�����<���;Yb�`������������`�H1:R���������؂>Ji�;&
�<h��<7�<�+T�]II=J�)���ٽ*��=�{������C>&Z�=UX;�X��㈽u6����=�mL<[Y=�v����"=�������=�.l=�k߼M ��^=��>",����k�y��jg�og=Q@���
=�e���<��2>'���X���x�����^�p�(�U=ㆢ=�C[=���^}:�FԄ=��>A{<���;��(�V�;�;K>��<��T=�!==�	=krQ=���;UR.�K�=�<��	�I�9<7�ӽ]��=���p47�t�F�l	�����N����=��{��t�=�#<��x��+���&=����^��<�1��\@�<�����;�n����Z>v=��
�����?�
>�5=W�;��=}��;><��Y%�=�3�<Ƽ>��E=n~ӽ�R�����*+��j:�/������9��=�0�:�Ⱥ�K�Ľq_
=�Я��i���v=����3`�=�3������d\��`�=�E:`�=A�=@|H=�o=���h���1�=�^5<�l�=��r<���������<��%=X��/�=^�<=��};q&|���Խ�����=':��4_�>i��YB�<ƈ=h����=.⊽��K��f���]�<���<K�=.�!:��?��=*8=�z=�zڼ���<�#E����<(���D���iѓ<�)�<Fx������4�
����=۰
���|=�u�;6Dƽfe�=)�.=}�d�{i�<@D���0ɽ����C�ν��#��=�O'=jU[=�+���_����d�w�=Zy�<�)�=W�G�	H�=��Ͻ�=t����}���f��3�6�t�=YA�=�t=��̼62>}>���=/�$>�8�s4=����Ž��^�O�y�v3x=�'N=J<��.Tͽ�p�=��=q<!�=Z�=��>��X� ꤽ��^=U�">�=O�=��&���q=�<���=���=j0>�=QG�<���.�̼u��=��Ǽ��8��P��B�;=�ջ��_<��>����WR�~�/�lh˽C�%<|�����^�:�#�=E��<�e�<GT�=���?M�=!:^<���<�c���d>�� =i\w�MDr=�"��Ӎ��=޽�=�4.�����5�=�jܽ�"^=��=���=����tO�=�4=�
{����=����o�<`��=��ӽH�v�_ep�.�>a�ûf@����H<�%K��l�<�Ax=�D}�8��9=��4��F���9:>��=X�=���R:=���^ϼ��:�y�����껍彵��.n=�߬=]��<w�=���(>99ν'��;�B�<F:���(��z><�Ĕ=�!��f��^="+�u�~��TB�,����G=Zs�<���2;�BG�����L<}�=�R�=�I;;�:�=<o���t=='~���y�=#s�l�<$N��J�<�l�� �c=�0�z������ļ>�=b�J>v�>�	��1X����=�Ý�_��<��=Г�<<��="��=1*�<&����<�K.=�8ܼ���=�Y�=k�2<���|j<?.½M�����<����ۄ<�ɓ����s�
���¼�A�<�ɩ�`���^B���B���2=��>��μ�F��g�=ͼ�<=�ĥ=J��ڽy��>.<=�j9�=�R�;o���ڼ_�p<~��V<p��=u���"��(2=	=�w���/���ڼ΂�n�6;Pz��$���ż��=��=%�a��=q�Ȼ�o��dN����̹����p���>��m=!��ڢ>���� *�e�=|<:=��=��=��V<�.M=cH^=�����8�ܘ�=����|��,�<A��T���N�,=����ݎ�<G����A��<
�:Rn��O?�uO�;2�=�2*��J��
��=BЭ�.׮<G�2=m��<�B��$O�<EBT<����
�k�d�r�E=�<�<GY`<�C�<�0=g�f�0F<���=ǚ����=W�����a={�<lj�P�r��:ǽ��T<_T�����i�=꙰�V=�=�ا=�mf��?]=�H�=S���X��{A�<i�=���<;u�l�&�
ꪽ�=<)�=_��Ff�=�AֽJ��=,xp��ʐ�q����#�f<�����x�=���S�=7	v�j�q=��H�i<�GG���������?�༫�S<)u=!o[<^\����=�Խ�f=%���y=�.<��=L��=������^k�=���_��Aк��Or=Z�߽l�߽�<Ti\>�'���6�;�)E����k}P<�r
=� ��eꍽ���8�<�h����<-&,;�=ϭ<+�=FR�����
�A=��=�V�J�c<vP�=��=����;��m=�g��p}1����tM��xp=pc�L���G�B���렼u�#>3�>}��:0,>�z�=Ȕ�I"̼I�~�lA >q�Y=���r�;��y��Y>��O���C=�j�<|��=t���$���$<D��=��=��썽�Y�Q.>{��	��=���6����<��D��	��t�H=K�W�����J�ĽmSI��]=;f	<��=
ⱽ9�<�kD>E,;W��U:~�T=^�-<�8�|@���	=l��=���=]�=]�	=�E��JA�=3u=Ә���-���1�=��<E��<�RƽH��=*��=�2½ӡ�EB+�f�X�e)��c<�	��.�<�׽R�W���q����=��;b��=��R�.�B<��R<�M< #�<� >�$`=�,=J�=��6�4�;)Ğ<�fؼ@=`�Z�d=�  ����=C<���=���=��\=�l�Q���0u��ý�J�gA�=m㉾�R�<�����g���6����$˽�v[�_��&�L��Tk=4����q=�%ۼ�ɽ���<z��I�ͻ����=�<�l:=w��<�'̼�Nz=�[1<d��=�qü .������Ʋ�ސ����=�k|=vϊ�0ת=��=�F<�!=��}<1���#�t��~7�A7�<�?�<G8�!�9=ذ��=<������d����f`�
��=dWR������=�O�=uMN�������=�1�)k���9y�0�S�jo�=�뾽X����<��߽�-��-�>��u=�-k="̈���=	�> ���l֚<��v=��=�±:U)Ǽ�+�<�/<�7qķ^r˼��Q=ρ��w->P�=����h'=HV=�;=��ýq�<lػ�m��i=������=�6Z<ir;�A=�k>X�O=���;���G:y���ļ �������p^���:�⾼|?��?pe=�ܿ=I&W���=�=
�pJ3�v��=��>�(A���=ǋ���Ͻ�I����G���?�g>��>�@㽛Ƭ�V� �|w4�v$W�<�\<U�=I_�=��<�&5=�P �e����Ը���=_=�J�=��#��r=ʽ�=�%�h������ֽ���=!�����8=t(Z<r�5>��<��������^��K�<z@=�M=�}G=l[�=$�ʽ}���)���=��=�ʿ��"�=�0Ͻ�u���L=s�J=�p���t�=��e=:>[3 >s���MA�g�[��K�=i���湽6v5=�0;kI�=	J=�v�;��;��<��T>_>��ICϽ�x=,�U�N�(=�̄��*�<+콭�׽����$8��"�����=9/�����,Z=5��=u|�<�E����M�l�M��+�N��K�Qڄ=Y_ͼŻ�=�[�=��=j$<x-�:?�=��һ2���Ǿ�+h�����[�A�z��<�ڽ�6=p�'=J��<� ;`��6�	���4�`��ŝ&��*��� ��笽���$�:���I��]4>[�=1�=wG>�-�Bܢ�] �=܈�=NĽ�L ��L�*r;��&����<�=J��۽Hh�u T=6��F�=Z=Ev	�Z%O=��]=Τ=>���Ya"=����_�<��ӽ-;ߥ�=�(����,<�>|�>�1ҼY$����O�&�<�@��TC�=fG�=u�=��l=fF��e��]��;���<�����_���=�r��⢽�(<:r</�ż��J�<P��=�4�R��=>$�ZҼ8�i<���쵫��(��<ۂ�� +<6}�<:��;��=�d(=����E<g�ܪ�=բ	��˽���!޽�T��Z^�=�R=P>���f=s�f���>��=^4�=
<��<�h���{���/�<*s`=�08�Y��O���n���꼺��=�%��
��y=E�=b��������-��0��s�#׽��
F��YМ��d�<Y�=�Ľ��
<���:�^'�|B��/����<܀��o���\�=�N�=�p��7��:iT��oP�d������.�=�={9�=�e=#]��9`�<M�q�����l���=*G���}@=1¼���G�L�=�=���cռU��5A�g�ǽ�O����s8><��N���[���9=��|����<m�;$���g#�*D^>����H����=�lH�.�Ҽ�8˼�0�u��=�k�<���v��<�����q>��P��M�R�ռ�`���=Y|�<X�=���=��=�oh�.�@<���=.b�Ol�=Y*�<j_�=����e-����=󁄼�,X=hp=5!/���1=���= �!�%-J=�=~�*;�H�=���hzB=3�?�
㚽�L	>sԼ�#ļ��>��<�x���=tfu�-�R=mb��g<��C�tm�<�+�=W�𽑖�� ^ּc޽z�=A̺�E.9���Y���x�.
��mK½�=���������1�=�N>'y�h>p��"�b=z��;S>�&����k��<H<v�'�]'�=���!i�=��*=ܨ=iiU��`�=]z=&�=?-�=��n=pÃ�#�b�`L]=�o�E��p��,�<�F�!�
�Jy����==�`;��=e��u���r�<��=x�1=����1V=��>a"!>����?G����X�����
J<�=���t��=MS�;]!�=>��Z��M����=�(��|R�Y�A�g@<�7��re�=��S=��:9�<f�=� [='��=$�=#6t��j��`�ڼr)����� P�M���ͻ��������Խ}����{<Ȁ��9�<����[��gL�(>:��~텼�Z���"��g��/�V�n��=�j=�9G�v�>�,�9�Z>�B��D�8>i�X�J��Ԉ��ȼ��½:�=N�\�4�K�w)��%��΄�2=*z�����䨽���VS=�jx�vF)=󑴽���=0o�<��=�/C�x���BE=^[�=~RM=�7��>�=�7>p!�=�_����=�h�φ�����:�|=N�;|��=a��sF�ub��x�=�oмm�ͼ��{=K�&�V�S<t��=]=�p�����f�=����o=���M�!�0�&=��нb@>���̅&��/��(���84==>q�r�f��<�	=I�����������=Z>��S=~f�B��;�?���j��d/x����'�踄�ipѺ�
@=��^�M�=��b�F���S	j<bW>{��<��<�r;]w�����+�(�Z��=l���	�=�K�<���OO��d��;��o��ɡ=n����ϽQ����=��޼d
������W�#�\���B�<�;>���<|H��~��1�;=SV����<&>��Լ�@���K����o�)��=�;<=8̽�"	�N<=��/�mO��ȍ<�ˑ�|�c<�z����>$y�;]���Q���"�<	Uܼ��4>J��=f��=!����.�=�'�=�n8,��>�-�=�O��n)�EY�=\n�=�=�<w��0�0=����Y��L�,=��=�>����������Z=Yϼ���<X!�;�&�<&н��0=�#O=N��=��+=�w�=��V=���=0�$�c���Ҿ<�.��T�=�h�<��>w������/�߽kF�<C+�=΃&=��>��.���%7;�H����;��νcZ�<xQ$=��@��aI�����?x�߹#=�Uf=�U=�km�2;?��_U�+�<��K���E;��4��Rǻ�BQ>��/;�&>����+y;*���[y��c��=U���ު� �s�.΅<�F���K{���C<�����Xͼͥ
=fB��N��[잽�
����j�X����>29���<,Fk����=S��#����=�a���^3= �.>[��<���=�ƭ��;�S�Z=���=�@r����=+�i������=���=AƢ=�Ѕ�a��=�~~=<�O={������/�k��<�0�=�#s�&�;�A>P1�<;ﺽ�!��P����>�<�o8<�齈�˻��e=�d�<Y4����:>�s�����.
��<��=^��=�D%��fS�8��<w==�x"=ڏh<�!�9
��;��=77J>e�	=�h=d4�B��=�Y^=f����սV�v�y佱�N<���eΒ=:��<ayI>$�>=��=�	���ͽ�䨼P͞>h��==V=��0���4����-RO�wsj��m+�r4�=0,~=5�!= �=U�;gU�<SD������N�[���<*�=��^�J��l	���L�|�B��j��<X�K=���C�w=W��<d��=������<3�=��<F�w�`ݻ2��=��ǋ�9�=����N(�<7���{J�oE*���}���M�}�e��C�=M��<{�=/�=����E���t<FR��e���������ؽ(Е=��;�6���=#�Q=�G��ʼC=��=�d�Qȼ����󊉼���mS��_=:�)"�=�:����C"&<5Q@�4�޽R�@=~]��7��=��*>޾g�w�����<�]˽���<�p���4 =�>��/<.V�=7�^=Ŭ��-O=���؊�=������� ��F�9��Ú�\�*�V���&�H Խ�;(Cw��| =rƷ�$���ȅ��w8���>��X�i,:=��o����=-&�=��?�W��T�0�O���O=>X�����p�!=	`3��U=���t"=J����ȼZg.�VoZ��0>�W=��<��=�I�=58Y���9��ﻓI=W�'��^��ܥ��T�3;���=���=�yh�5���K��=)n=�3����=�U<.���Mq�<0�L�[���q�=T��=���]�;�U=�K�="�>A����������1����7�6�6=��f�I�¼j�u=�p����=(w�j�н̽�x\=��=O�=oyc�X��=[����=����;��)�׼�=�m��թ=��}<!��_4��>��"�lɇ�� =�����:�rI�<~+���ĽQ�ؽ����A���胼�#=K2[��≽L����:��E�=DgR=gkĻ�=� ;�4=��۽u�����<C7n<GjG�Uj�<S��M���$b�=�л�����_�5p�����=�z	��l;u�����=�n=�Cw=�"ٽ��=��F=M��<��a=f�&�D(�=5�ƽ���<�'�=���=��-=_��==�>���g >f�Q�@���U�;}%H=�� �n;	�����@�=�P��}�|����B�ڮ=���=�\=�p�
��<N�ֽ,ڟ��=��9=�o?<Ad��Ǚ��Ǽ8��?�=��=��=@4�=�4�=-K>@��P��"+
�����/=�Jf�������=&a���!�hm���7=)�/=��6�b��<�����?=E�l��F�=�>�<�`�=+ZP<BC�=qF��Ww��� ü���=�\@=?Uh<�>����5���;�=��F�L�6�{6��s[�K�]��$�3�C�5s�=F�����=���;�1��ݠ=��&<��ɽ�(>��B���n��"�=S]�=��u��� d�<~D$�;�9�2���4�x<�B��߃>z1��#=�#>�o>G�����]H:���b�8{�=�:�=�-=q�=gS�z�	=:(>�,�Z�q��T=�Z>��c�Y?��н~3�0�ϼ<o;a�H��Ž�MR=;ýB�ż�75=~Q=���*�=~#��s5�=���=�q-=���zV4��4D��w<���o�=7/�<�� =�k-=�<!��>��;��=�X��t9��j;<�t켏�(>����6H��=�s��="K=�3���������<r�Q=��<�?9���=b�w��m �gY�����_����9?=���������8O=���=�(E��M2��6>�;�=��$���8����΁�����:��=r�=��U=���	>�=P=����,��#�{�����k75�ͧ-��,=֢ͽ�)e=4���@��;��u=�/��� ����*>�������<Հ��:3��ńU��w��2�=��X=<Hq=�<�<��NJ� o�X;����6�=:9�<\.6�t�j�7=��=�[�<n��;�w���=Jr&=C�>��KT�אK=w��=;>9��<'@��;�=��3�`B��2���H�O<��>E���U=��ջ�����	�2O�<��=��<�)���c��������=?$>0>c�=�<�	C=��޽5Fm�Æ=I�=����Q�ü��<PH5=�^<�0=3��=��8����������(��=�E�=3���?H=��x�㘷<�t>�T=�K�eVs�<�>é�<�U51�>�fv�� ���Ik�m8�=�f-���A=3D=> a�=vK��sʹ��V�=x�=�3h���ᾠ=��ͼ�I�="VD�k�s=*S
=�8>����x���I���u��0G�<�A]�϶ <ʈ���=P�;�OټZ�=}�J=�\5��HS�d�>=��=R2�<�Ja=�����0�Խm��D�(<�oǽ5-�<�"=��5=5�<�ީ=�Q{���m��6=J% ��d�=}����C)>��M�
�2�)[�<Ӂ =�W�o����?-<����g=��=}�.�[m�#i(�c	[=Z?��m�㽧�ü෯=Q%�=zȼ���=6�����B<�?<jHF>�S�2�=�NR<��>����3-�[�	��oH�.����v���=��~�'@��/	��bнB�_=��=q�=�=�=.�½W�P=h8�=���<5G�	�`=���=0t��D�=�ϼ���ƽ$=J�>URK�ސ�<<���=���?��<�N��q�E]G�ڭܽyj5=�[��z�=m�<
Z<m}`<�<7�7��d���=墍�z������=�a����=���=Y��=D@��8�ɠY=��k=�8��D��<Z��5E�<�z�<�c+��<R=	� �Q(�n�	�m<�a!<�]����>/��\��Qe����6���p=&=��̹����<=P5Ǽ�*=٪>B�=���=����L���3�=?^�.����ýV)�)\ʽ|����<*2�:�_)�d��=� I�)X�$�����]��k���<�=���=]�R<$R=�����?�N�=\��7�E���l|*�JƼ����=�������<���=U/X=����� >ϰ���>���}f����=3�ռ|p��4���{ �!��&�&�
/�<):\<'�=�=e=]`*������:9<��>���=�1�5�3����� �;�r=����'=��
=w$z��@=�#�����c<�;���G>������=��?>�=+������=F�޼�<> ����5=~z��.M�?�=��<$�'=܄� �=�7u=�P@<�ˍ��W��f�S�.>��S=�{�<�$ �:)����#���	<�ź�G=)!�]=/f��T�o=w�=�9���&0=�dU=+)�=��J���lE��փT����3<=�>��W>��罄x��b<�=�>���{���Q=���$�)�/����<4_�=Y2ώ�.���U=�.���>�Ky��i߽ݦi�lA=�0>cV*>��JcM=�
{=��>C�L=����ȼ���<<�<qJ�;�6�=�m=�8J�9����=w���Z�=�9)=/������+��̙�=�tz=*� ��>��<�	��܊�4��=���=m�߽�#�=�P=O� >yt��ꖼ.u�=�Qq=��"=`PU=Z��o�����(�=�g�q�%=M�b=���x�z�f|�<����#>=�ӌ=��ܽ��F��a?=+�������̼ؒWVv=��;W�%�!�=$r;]ӈ���X����=P͈=�
9��s=>�ν&�"><=;ӝ�`��=?>�<�0�=r��@��&"�Zg�<㞽ؚ=4�g;2�=��>��=	%�<c4���-n>%$B���D��D�9u:`=yoB=*p�K)��s3��d�=��Ͻ;̋=��w=��=�JL�!*>x��<*���%=N���=�h�<V�=z0`=Nǣ�����;H�<^┼���Л9=v_�����_�;��������Wq�,gӼ��=�^����I�(�*>K��=_����Y�-ϕ9o�E=�ٽl=/��:��=U�9���>ؚ����⼈&]=m�w��:>qp���J=�䎽�}���=(;�6����<݄�;%�89���<M�y͕<�n7�� =�.\�X��;� =�fD���D�w%K�O��=�赽ɸ�	-��7*� �K��s�-dѽϭ$=|�=	Ľ����Խ���~=�P����*�8��=t� ��Q=���<�S��h{𽑻	��v5����=C�۽"�&=W�J��;H=<�;�8��=Cv>���=����n=k�����=y�"�E�=�|���d4�è=FA=/�>O_=�eǽU7�=��> �=����.O=^�<�[U�.��=��>�X�6�e<�L�:G�=7��������̚!=��*>� 3���<�<_���e���=���=�/�<>i��</���
ۼP %=Rָ���8�'�&= ŏ�k�=���=.>����=a�м�SZ<�(=�L����μ/�<���<ߐ'<=�#>���<�=�(���+�J���9�<$��(��<]o!>	�M�/%��@D=#3�=Rh0=�)=W`�;b�>ì��]������<ؽ�v3>>\=P�=�༹���CΙ=������[�}������=�)#�?<'>��Cz�������]����=����H=@��=���<sѬ�I��
)��|w<ף=�˼}6�=a	��r�= Ľ����=�Z���<��w��T�=<5�="��<5-!=(�A��=6�=d:�a`7>�z�=�>#XJ<�ā��@�-k�=蟼�(���lP�ǸH=��<�5�=���38|=��=`_ʽ5��	->|=�Q=��=��:�,Q�<�rK��`�=S�O=��=��F��Qɽ��3���,�9�=��<Y�������n�=↔�_4���5�x޸=�vý|�=��׼��ؽ$Q�U"���G =#&�=L��<�e!�_����
��Tf��:^=���=ՠ��t�<�����c��A=�\o���=��=<ҟ<"�Q=�&�<Ld��*�\����=U$�=ȼk9��S>��=4�;�u<"Fb�#�=�Bϻ�$f��s�=v�����<ߕ����1=�o�����\�=)q2���%.Ż��ü���<�9��&½">�U��o�c=,�L=��-=��=8>��D<�̾�|h=%�y=<��<���=}�=�F�=��=P �З��Ĺ�HQ��V��<x#�,=�@I=��=e�Ӽ4���mڽ���/�����=g��h@:�
މ=h����*=r�-��.>�ٽ3R_���#��>��;��]=�66�z_�<���=�Rd���=g��;�]>y3��5�;�᥽��<�V��~D���=�s�<~ ��ua�=/�=�O�=�`~:��4�p����=�-=_���=�(���~���֮=�7=K�佒6{�C�
�o1۽�h��[]��M>08�=H^>Dt�E��<�$@=�ڼ}㱽� >+V�=�/>*
_output_shapes
:	(�
�
lstm_1/recurrent_kernel
VariableV2*
dtype0*
shape:	(�*
	container *
shared_name *
_output_shapes
:	(�
�
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes
:	(�
�
lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
T0*
_output_shapes
:	(�
]
lstm_1/Const_1Const*
dtype0*
valueB�*    *
_output_shapes	
:�
y
lstm_1/bias
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/Const_1*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes	
:�
o
lstm_1/bias/readIdentitylstm_1/bias*
_class
loc:@lstm_1/bias*
T0*
_output_shapes	
:�
]
lstm_1/PlaceholderPlaceholder*
dtype0*
shape:�*
_output_shapes	
:�
�
lstm_1/AssignAssignlstm_1/biaslstm_1/Placeholder*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking( *
T0*
_output_shapes	
:�
`
lstm_1/initNoOp^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^lstm_1/bias/Assign
k
lstm_1/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
m
lstm_1/strided_slice/stack_1Const*
dtype0*
valueB"    (   *
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

:(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_1/stackConst*
dtype0*
valueB"    (   *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_1Const*
dtype0*
valueB"    P   *
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

:(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_2/stackConst*
dtype0*
valueB"    P   *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
dtype0*
valueB"    x   *
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

:(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_3/stackConst*
dtype0*
valueB"    x   *
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

:(*

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
valueB"    (   *
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

:((*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_5/stackConst*
dtype0*
valueB"    (   *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_1Const*
dtype0*
valueB"    P   *
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

:((*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_6/stackConst*
dtype0*
valueB"    P   *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
dtype0*
valueB"    x   *
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

:((*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_7/stackConst*
dtype0*
valueB"    x   *
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

:((*

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
valueB:(*
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
:(*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
f
lstm_1/strided_slice_9/stackConst*
dtype0*
valueB:(*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
dtype0*
valueB:P*
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
:(*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_10/stackConst*
dtype0*
valueB:P*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
dtype0*
valueB:x*
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
:(*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_11/stackConst*
dtype0*
valueB:x*
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
:(*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
b
lstm_1/zeros_like	ZerosLikelstm_1_input*
T0*+
_output_shapes
:���������
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
valueB"   (   *
_output_shapes
:
�
lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*'
_output_shapes
:���������(*
T0*
data_formatNHWC
a
lstm_1/stackConst*
dtype0*!
valueB"����   (   *
_output_shapes
:
}
lstm_1/Reshape_1Reshapelstm_1/BiasAddlstm_1/stack*+
_output_shapes
:���������(*
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
:���������(
�
lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*'
_output_shapes
:���������(*
T0*
data_formatNHWC
c
lstm_1/stack_1Const*
dtype0*!
valueB"����   (   *
_output_shapes
:
�
lstm_1/Reshape_3Reshapelstm_1/BiasAdd_1lstm_1/stack_1*+
_output_shapes
:���������(*
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
:���������(
�
lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*'
_output_shapes
:���������(*
T0*
data_formatNHWC
c
lstm_1/stack_2Const*
dtype0*!
valueB"����   (   *
_output_shapes
:
�
lstm_1/Reshape_5Reshapelstm_1/BiasAdd_2lstm_1/stack_2*+
_output_shapes
:���������(*
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
:���������(
�
lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*'
_output_shapes
:���������(*
T0*
data_formatNHWC
c
lstm_1/stack_3Const*
dtype0*!
valueB"����   (   *
_output_shapes
:
�
lstm_1/Reshape_7Reshapelstm_1/BiasAdd_3lstm_1/stack_3*+
_output_shapes
:���������(*
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

Tidx0*,
_output_shapes
:����������*
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
T0*,
_output_shapes
:����������
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
Index0*(
_output_shapes
:����������*

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
valueB"    (   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_15/stackConst*
dtype0*
valueB"    (   *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_1Const*
dtype0*
valueB"    P   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_16/stackConst*
dtype0*
valueB"    P   *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_1Const*
dtype0*
valueB"    x   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_17/stackConst*
dtype0*
valueB"    x   *
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
:���������(*

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
:���������(
�
lstm_1/MatMul_4MatMullstm_1/mul_1lstm_1/strided_slice_4*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
o
lstm_1/add_1Addlstm_1/strided_slice_14lstm_1/MatMul_4*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_2*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/MatMul_5MatMullstm_1/mul_3lstm_1/strided_slice_5*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
o
lstm_1/add_3Addlstm_1/strided_slice_15lstm_1/MatMul_5*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_4*
T0*'
_output_shapes
:���������(
j
lstm_1/mul_5Mullstm_1/clip_by_value_1lstm_1/Tile*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/MatMul_6MatMullstm_1/mul_6lstm_1/strided_slice_6*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
o
lstm_1/add_5Addlstm_1/strided_slice_16lstm_1/MatMul_6*
T0*'
_output_shapes
:���������(
S
lstm_1/TanhTanhlstm_1/add_5*
T0*'
_output_shapes
:���������(
h
lstm_1/mul_7Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:���������(
a
lstm_1/add_6Addlstm_1/mul_5lstm_1/mul_7*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/MatMul_7MatMullstm_1/mul_8lstm_1/strided_slice_7*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
o
lstm_1/add_7Addlstm_1/strided_slice_17lstm_1/MatMul_7*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_6*
T0*'
_output_shapes
:���������(
U
lstm_1/Tanh_1Tanhlstm_1/add_6*
T0*'
_output_shapes
:���������(
m
lstm_1/mul_10Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:���������(
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
:���������(*
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*
parallel_iterations 
�
lstm_1/while/Enter_3Enterlstm_1/Tile*'
_output_shapes
:���������(*
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
:���������(: *
T0*
N
�
lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*)
_output_shapes
:���������(: *
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
&:���������(:���������(
�
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:���������(:���������(
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
:���������(
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*
T0*'
_output_shapes
:���������(
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
loc:@lstm_1/TensorArray_1*(
_output_shapes
:����������
�
 lstm_1/while/strided_slice/stackConst^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:
�
"lstm_1/while/strided_slice/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    (   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_1/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    (   *
_output_shapes
:
�
$lstm_1/while/strided_slice_1/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    P   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_2/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    P   *
_output_shapes
:
�
$lstm_1/while/strided_slice_2/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    x   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_3/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    x   *
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
:���������(*

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
:���������(
�
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice_4*
_output_shapes

:((*
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
:���������(
z
lstm_1/while/addAddlstm_1/while/strided_slicelstm_1/while/MatMul*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_5*
_output_shapes

:((*
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
:���������(
�
lstm_1/while/add_2Addlstm_1/while/strided_slice_1lstm_1/while/MatMul_1*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*
T0*'
_output_shapes
:���������(
�
lstm_1/while/mul_4Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_3*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_6*
_output_shapes

:((*
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
:���������(
�
lstm_1/while/add_4Addlstm_1/while/strided_slice_2lstm_1/while/MatMul_2*
T0*'
_output_shapes
:���������(
_
lstm_1/while/TanhTanhlstm_1/while/add_4*
T0*'
_output_shapes
:���������(
z
lstm_1/while/mul_6Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*
T0*'
_output_shapes
:���������(
s
lstm_1/while/add_5Addlstm_1/while/mul_4lstm_1/while/mul_6*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_7*
_output_shapes

:((*
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
:���������(
�
lstm_1/while/add_6Addlstm_1/while/strided_slice_3lstm_1/while/MatMul_3*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:���������(
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_5*
T0*'
_output_shapes
:���������(
~
lstm_1/while/mul_9Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:���������(
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
:���������(
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/add_5*
T0*'
_output_shapes
:���������(
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
:���������(
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*
T0*'
_output_shapes
:���������(
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
 :������������������(*$
element_shape:���������(
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
:���������(
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
 :������������������(
e
lstm_1/Square_1Squarelstm_1/TensorArrayReadV3*
T0*'
_output_shapes
:���������(
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
:���������(
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
valueB"(      *
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
valueB
 *�þ*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
dtype0*
valueB
 *��>*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2���*
seed���)*
T0*
_output_shapes

:(
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:(
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:(
�
dense_1/kernel
VariableV2*
dtype0*
shape
:(*
	container *
shared_name *
_output_shapes

:(
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:(
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:(
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
:���������(
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
valueB"   �   *
_output_shapes
:
�
gradients/lstm_1/Sum_grad/TileTile!gradients/lstm_1/Sum_grad/Reshape(gradients/lstm_1/Sum_grad/Tile/multiples*

Tmultiples0*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes
:	�
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
:���������(
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
:���������(
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
:���������(*
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
valueB"   �   *
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
:	�
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
:	�
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
:	�*
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
:���������(
�
$gradients/lstm_1/Square_1_grad/mul_1Mul&gradients/lstm_1/mul_11_grad/Reshape_1"gradients/lstm_1/Square_1_grad/mul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������(
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
:	�
�
"gradients/lstm_1/Square_grad/mul_1Mul#gradients/lstm_1/mul_grad/Reshape_1 gradients/lstm_1/Square_grad/mul* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes
:	�
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
:���������(
�
&gradients/dense_1/MatMul_grad/MatMul_1MatMullstm_1/TensorArrayReadV3$gradients/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_1/MatMul*
T0*
_output_shapes

:(
�
gradients/AddNAddN$gradients/lstm_1/Square_1_grad/mul_1$gradients/dense_1/MatMul_grad/MatMul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������(*
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
:���������(
j
gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_3*
T0*'
_output_shapes
:���������(
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
:���������(*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_2*
is_constant( 
�
)gradients/lstm_1/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*'
_output_shapes
:���������(*4

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
:���������(: *
N
�
-gradients/lstm_1/while/Switch_3_grad/b_switchMerge)gradients/lstm_1/while/Exit_3_grad/b_exit4gradients/lstm_1/while/Switch_3_grad_1/NextIteration*'
_class
loc:@lstm_1/while/Merge_3*
T0*)
_output_shapes
:���������(: *
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
&:���������(:���������(
�
*gradients/lstm_1/while/Merge_3_grad/SwitchSwitch-gradients/lstm_1/while/Switch_3_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:���������(:���������(
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
:���������(
�
gradients/AddN_1AddN,gradients/lstm_1/while/Merge_2_grad/Switch:1Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:���������(*
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
:���������(
�
%gradients/lstm_1/while/mul_9_grad/mulMulgradients/AddN_1.gradients/lstm_1/while/mul_9_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
�
'gradients/lstm_1/while/mul_9_grad/mul_1Mul0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopgradients/AddN_1*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:���������(
�
4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:���������(
�
<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(*
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
:���������(*
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
:���������(*
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
:���������(*
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
:���������(
�
%gradients/lstm_1/while/mul_4_grad/mulMul)gradients/lstm_1/while/add_5_grad/Reshape.gradients/lstm_1/while/mul_4_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
�
'gradients/lstm_1/while/mul_4_grad/mul_1Mul0gradients/lstm_1/while/mul_4_grad/mul_1/StackPop)gradients/lstm_1/while/add_5_grad/Reshape*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
�
%gradients/lstm_1/while/mul_6_grad/mulMul+gradients/lstm_1/while/add_5_grad/Reshape_1.gradients/lstm_1/while/mul_6_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
�
'gradients/lstm_1/while/mul_6_grad/mul_1Mul0gradients/lstm_1/while/mul_6_grad/mul_1/StackPop+gradients/lstm_1/while/add_5_grad/Reshape_1*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
�
%gradients/lstm_1/while/mul_8_grad/mulMul)gradients/lstm_1/while/add_7_grad/Reshape.gradients/lstm_1/while/mul_8_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(*
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:���������(
�
4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
4gradients/lstm_1/while/clip_by_value_grad/LogicalNot
LogicalNot6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:���������(
�
2gradients/lstm_1/while/clip_by_value_grad/Select_1Select4gradients/lstm_1/while/clip_by_value_grad/LogicalNot)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
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
:���������(*
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
:���������(*
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:���������(
�
<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot
LogicalNot;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:���������(
�
:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Select<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������(
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
:���������(*
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
:���������(*
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
:���������(*
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
Index0*(
_output_shapes
:����������*

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

:((*4

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
:���������(
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
:���������(
�
-gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_6_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_3*
T0*
_output_shapes

:((
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
:���������(*
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
:���������(*
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
Index0*(
_output_shapes
:����������*

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

:((*4

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
:���������(
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
:���������(
�
-gradients/lstm_1/while/MatMul_2_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_4_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_2*
T0*
_output_shapes

:((
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
:���������(
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
:���������(*
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
:���������(
�
'gradients/lstm_1/while/mul_7_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_3_grad/MatMul*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������(
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
valueB((*    *
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
_output_shapes

:((*4

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
:((: *
N
�
1gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*(
_output_shapes
:((:((
�
.gradients/lstm_1/while/MatMul_3/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_3_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:((
�
8gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_3/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:((
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
:���������(
�
%gradients/lstm_1/while/mul_3_grad/mulMul)gradients/lstm_1/while/add_3_grad/Reshape.gradients/lstm_1/while/mul_3_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(*
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
:���������(
�
%gradients/lstm_1/while/mul_1_grad/mulMul)gradients/lstm_1/while/add_1_grad/Reshape.gradients/lstm_1/while/mul_1_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(*
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
:���������(
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
:���������(*
Tshape0
�
'gradients/lstm_1/while/mul_5_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_2_grad/MatMul*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:���������(
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
valueB((*    *
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
_output_shapes

:((*4

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
:((: *
N
�
1gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*(
_output_shapes
:((:((
�
.gradients/lstm_1/while/MatMul_2/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_2_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:((
�
8gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_2/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:((
�
+gradients/lstm_1/strided_slice_7_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_7*
valueB"(   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_22gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes
:	(�*

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
:���������(*
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
:���������(*
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
:���������(*
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
:���������(*
Tshape0
�
+gradients/lstm_1/strided_slice_6_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_6*
valueB"(   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_22gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes
:	(�*

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
Index0*(
_output_shapes
:����������*

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

:((*4

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
:���������(
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
:���������(
�
-gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_2_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_1*
T0*
_output_shapes

:((
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
Index0*(
_output_shapes
:����������*

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

:((*4

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
:���������(
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
:���������(
�
+gradients/lstm_1/while/MatMul_grad/MatMul_1MatMul4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop)gradients/lstm_1/while/add_grad/Reshape_1*
transpose_b( *
transpose_a(*&
_class
loc:@lstm_1/while/MatMul*
T0*
_output_shapes

:((
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
:���������(
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
:���������(*
Tshape0
�
'gradients/lstm_1/while/mul_2_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_1_grad/MatMul*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:���������(
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
valueB((*    *
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
_output_shapes

:((*4

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
:((: *
N
�
1gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*(
_output_shapes
:((:((
�
.gradients/lstm_1/while/MatMul_1/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_1_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:((
�
8gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_1/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:((
�
gradients/AddN_3AddN<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad:gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*(
_output_shapes
:����������*
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
:���������(
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
:���������(*
Tshape0
�
%gradients/lstm_1/while/mul_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop)gradients/lstm_1/while/MatMul_grad/MatMul*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:���������(
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
valueB((*    *
_output_shapes

:((
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter.gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
_output_shapes

:((*4

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
:((: *
N
�
/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*(
_output_shapes
:((:((
�
,gradients/lstm_1/while/MatMul/Enter_grad/AddAdd1gradients/lstm_1/while/MatMul/Enter_grad/Switch:1+gradients/lstm_1/while/MatMul_grad/MatMul_1*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:((
�
6gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration,gradients/lstm_1/while/MatMul/Enter_grad/Add*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:((
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit/gradients/lstm_1/while/MatMul/Enter_grad/Switch*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:((
�
+gradients/lstm_1/strided_slice_5_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_5*
valueB"(   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_22gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes
:	(�*

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
:���������(*
N
�
+gradients/lstm_1/strided_slice_4_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_4*
valueB"(   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_20gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes
:	(�*

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
loc:@lstm_1/TensorArray_1*,
_output_shapes
:����������*
element_shape:
�
4gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_4*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:���������(
�
gradients/AddN_5AddN6gradients/lstm_1/strided_slice_7_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_6_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_5_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*)
_class
loc:@lstm_1/strided_slice_7*
T0*
_output_shapes
:	(�*
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
T0*,
_output_shapes
:����������*#
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
:���������(
�
$gradients/lstm_1/concat_grad/Slice_1Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:1%gradients/lstm_1/concat_grad/ShapeN:1*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������(
�
$gradients/lstm_1/concat_grad/Slice_2Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:2%gradients/lstm_1/concat_grad/ShapeN:2*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������(
�
$gradients/lstm_1/concat_grad/Slice_3Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:3%gradients/lstm_1/concat_grad/ShapeN:3*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������(
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
:���������(*
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
:���������(*
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
:���������(*
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
:���������(*
Tshape0
�
)gradients/lstm_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_1_grad/Reshape*!
_class
loc:@lstm_1/BiasAdd*
T0*
_output_shapes
:(*
data_formatNHWC
�
+gradients/lstm_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_3_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_1*
T0*
_output_shapes
:(*
data_formatNHWC
�
+gradients/lstm_1/BiasAdd_2_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_5_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_2*
T0*
_output_shapes
:(*
data_formatNHWC
�
+gradients/lstm_1/BiasAdd_3_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_7_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_3*
T0*
_output_shapes
:(*
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

:(
�
+gradients/lstm_1/strided_slice_8_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_8*
valueB:�*
_output_shapes
:
�
6gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2)gradients/lstm_1/BiasAdd_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes	
:�*

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

:(
�
+gradients/lstm_1/strided_slice_9_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_9*
valueB:�*
_output_shapes
:
�
6gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2+gradients/lstm_1/BiasAdd_1_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes	
:�*

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

:(
�
,gradients/lstm_1/strided_slice_10_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_10*
valueB:�*
_output_shapes
:
�
7gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2+gradients/lstm_1/BiasAdd_2_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes	
:�*

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

:(
�
,gradients/lstm_1/strided_slice_11_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_11*
valueB:�*
_output_shapes
:
�
7gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2+gradients/lstm_1/BiasAdd_3_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes	
:�*

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
valueB"   �   *
_output_shapes
:
�
4gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad)gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2%gradients/lstm_1/MatMul_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes
:	�*

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
valueB"   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2'gradients/lstm_1/MatMul_1_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes
:	�*

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
valueB"   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2'gradients/lstm_1/MatMul_2_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes
:	�*

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
valueB"   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2'gradients/lstm_1/MatMul_3_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes
:	�*

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
T0*
_output_shapes	
:�*
N
�
gradients/AddN_7AddN"gradients/lstm_1/Square_grad/mul_14gradients/lstm_1/strided_slice_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_1_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_2_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_3_grad/StridedSliceGrad* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes
:	�*
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
valueB(*    *
_output_shapes

:(
~

Variable_1
VariableV2*
dtype0*
shape
:(*
	container *
shared_name *
_output_shapes

:(
�
Variable_1/AssignAssign
Variable_1Const_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:(
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes

:(
V
Const_4Const*
dtype0*
valueB�*    *
_output_shapes	
:�
x

Variable_2
VariableV2*
dtype0*
shape:�*
	container *
shared_name *
_output_shapes	
:�
�
Variable_2/AssignAssign
Variable_2Const_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes	
:�
^
Const_5Const*
dtype0*
valueB	�*    *
_output_shapes
:	�
�

Variable_3
VariableV2*
dtype0*
shape:	�*
	container *
shared_name *
_output_shapes
:	�
�
Variable_3/AssignAssign
Variable_3Const_5*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:	�
p
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:	�
^
Const_6Const*
dtype0*
valueB	(�*    *
_output_shapes
:	(�
�

Variable_4
VariableV2*
dtype0*
shape:	(�*
	container *
shared_name *
_output_shapes
:	(�
�
Variable_4/AssignAssign
Variable_4Const_6*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes
:	(�
p
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes
:	(�
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

:(
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

:(
F
mul_6Mulsub_3Square_2*
T0*
_output_shapes

:(
C
add_4Addmul_5mul_6*
T0*
_output_shapes

:(
�
Assign_2Assign
Variable_1add_4*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:(
f
mul_7Mullr/read&gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:(
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

:(
e
clip_by_value_1Maximumclip_by_value_1/MinimumConst_9*
T0*
_output_shapes

:(
H
Sqrt_1Sqrtclip_by_value_1*
T0*
_output_shapes

:(
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

:(
G
div_2RealDivmul_7add_5*
T0*
_output_shapes

:(
Q
sub_4Subdense_1/kernel/readdiv_2*
T0*
_output_shapes

:(
�
Assign_3Assigndense_1/kernelsub_4*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:(
M
mul_8Mulrho/readVariable_2/read*
T0*
_output_shapes	
:�
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
J
Square_3Squaregradients/AddN_6*
T0*
_output_shapes	
:�
C
mul_9Mulsub_5Square_3*
T0*
_output_shapes	
:�
@
add_6Addmul_8mul_9*
T0*
_output_shapes	
:�
�
Assign_4Assign
Variable_2add_6*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes	
:�
N
mul_10Mullr/readgradients/AddN_6*
T0*
_output_shapes	
:�
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
Y
clip_by_value_2/MinimumMinimumadd_6Const_12*
T0*
_output_shapes	
:�
c
clip_by_value_2Maximumclip_by_value_2/MinimumConst_11*
T0*
_output_shapes	
:�
E
Sqrt_2Sqrtclip_by_value_2*
T0*
_output_shapes	
:�
L
add_7/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
C
add_7AddSqrt_2add_7/y*
T0*
_output_shapes	
:�
E
div_3RealDivmul_10add_7*
T0*
_output_shapes	
:�
K
sub_6Sublstm_1/bias/readdiv_3*
T0*
_output_shapes	
:�
�
Assign_5Assignlstm_1/biassub_6*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes	
:�
R
mul_11Mulrho/readVariable_3/read*
T0*
_output_shapes
:	�
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
N
Square_4Squaregradients/AddN_7*
T0*
_output_shapes
:	�
H
mul_12Mulsub_7Square_4*
T0*
_output_shapes
:	�
F
add_8Addmul_11mul_12*
T0*
_output_shapes
:	�
�
Assign_6Assign
Variable_3add_8*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:	�
R
mul_13Mullr/readgradients/AddN_7*
T0*
_output_shapes
:	�
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
]
clip_by_value_3/MinimumMinimumadd_8Const_14*
T0*
_output_shapes
:	�
g
clip_by_value_3Maximumclip_by_value_3/MinimumConst_13*
T0*
_output_shapes
:	�
I
Sqrt_3Sqrtclip_by_value_3*
T0*
_output_shapes
:	�
L
add_9/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
G
add_9AddSqrt_3add_9/y*
T0*
_output_shapes
:	�
I
div_4RealDivmul_13add_9*
T0*
_output_shapes
:	�
Q
sub_8Sublstm_1/kernel/readdiv_4*
T0*
_output_shapes
:	�
�
Assign_7Assignlstm_1/kernelsub_8*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes
:	�
R
mul_14Mulrho/readVariable_4/read*
T0*
_output_shapes
:	(�
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
N
Square_5Squaregradients/AddN_5*
T0*
_output_shapes
:	(�
H
mul_15Mulsub_9Square_5*
T0*
_output_shapes
:	(�
G
add_10Addmul_14mul_15*
T0*
_output_shapes
:	(�
�
Assign_8Assign
Variable_4add_10*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes
:	(�
R
mul_16Mullr/readgradients/AddN_5*
T0*
_output_shapes
:	(�
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
^
clip_by_value_4/MinimumMinimumadd_10Const_16*
T0*
_output_shapes
:	(�
g
clip_by_value_4Maximumclip_by_value_4/MinimumConst_15*
T0*
_output_shapes
:	(�
I
Sqrt_4Sqrtclip_by_value_4*
T0*
_output_shapes
:	(�
M
add_11/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
I
add_11AddSqrt_4add_11/y*
T0*
_output_shapes
:	(�
J
div_5RealDivmul_16add_11*
T0*
_output_shapes
:	(�
\
sub_10Sublstm_1/recurrent_kernel/readdiv_5*
T0*
_output_shapes
:	(�
�
Assign_9Assignlstm_1/recurrent_kernelsub_10*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes
:	(�
�
group_deps_1NoOp^add_1^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
�
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign
^lr/Assign^rho/Assign^decay/Assign^iterations/Assign^Variable/Assign^Variable_1/Assign^Variable_2/Assign^Variable_3/Assign^Variable_4/Assign"X9�_��     jዱ	�Z7DI�AJ��
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
Ttype*1.0.12v1.0.0-65-g4763edf-dirty��
b
lstm_1_inputPlaceholder*
dtype0*
shape: *+
_output_shapes
:���������
l
lstm_1/random_uniform/shapeConst*
dtype0*
valueB"   �   *
_output_shapes
:
^
lstm_1/random_uniform/minConst*
dtype0*
valueB
 *�E�*
_output_shapes
: 
^
lstm_1/random_uniform/maxConst*
dtype0*
valueB
 *�E>*
_output_shapes
: 
�
#lstm_1/random_uniform/RandomUniformRandomUniformlstm_1/random_uniform/shape*
dtype0*
seed2���*
seed���)*
T0*
_output_shapes
:	�
w
lstm_1/random_uniform/subSublstm_1/random_uniform/maxlstm_1/random_uniform/min*
T0*
_output_shapes
: 
�
lstm_1/random_uniform/mulMul#lstm_1/random_uniform/RandomUniformlstm_1/random_uniform/sub*
T0*
_output_shapes
:	�
|
lstm_1/random_uniformAddlstm_1/random_uniform/mullstm_1/random_uniform/min*
T0*
_output_shapes
:	�
�
lstm_1/kernel
VariableV2*
dtype0*
shape:	�*
shared_name *
	container *
_output_shapes
:	�
�
lstm_1/kernel/AssignAssignlstm_1/kernellstm_1/random_uniform*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes
:	�
y
lstm_1/kernel/readIdentitylstm_1/kernel* 
_class
loc:@lstm_1/kernel*
T0*
_output_shapes
:	�
U
lstm_1/SquareSquarelstm_1/kernel/read*
T0*
_output_shapes
:	�
Q
lstm_1/mul/xConst*
dtype0*
valueB
 *o�:*
_output_shapes
: 
X

lstm_1/mulMullstm_1/mul/xlstm_1/Square*
T0*
_output_shapes
:	�
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
��
%lstm_1/recurrent_kernel/initial_valueConst*
dtype0*��
value��B��	(�"����_���D<���=��;���=]Go��޽��I��-O�qQ��.�<�s�:�E��� =�������c�;�r	���,��OݽD��2��<=6��<�K>����6�7�nۤ<ٵ뽂���|���T=���<�o���߼=��o���=f�	=��<V��<{#��<?��5�e�iߓ����P���r��3Q9�����~}� Tz�8�!ѳ=�\�=��۽���L�ӽI�=���=�E�=W\=ɇ�=ް4=`�?=S�ݼtY��*�=����M�<Ӛ/=u�=z ¼s9=��Ͻ���HT�ڶJ��=U��<Bdؽ�/>��нT��<`%����ec�<�ϣ;圡<	��<�Q��)2=�rI<ঁ�
5d���̽1[��xm�7�=�7�=mW�=�M;���#��R)�l�,�o>+�j|��_�����=32{=w\K����E�n=��=2�!���<y�A���>t�w��s��*>��;���w��=�(�iW��6׻��ҽ�N�=��>�~=�νX?���M���=Q�.>#D=f�=qSR��=�h�=�����J=>�^<���E`=.ݻ7�s=��c�����'Ծ��ι������
=d���*x�I\�=�=��m�Z�@_a���=��=�7��2>�=خ&�[�ٻ��c>��k��
�=?��;�ө��u�=�#Ὠ1��w~�������=#;�<bR��p<�����B�=ӳ,��c�<U��=}�a� 4|<��}�ؒ�=�u���=��K:ݖ=��Ľ�0���\�=)ܜ=?z1�A�<=�ΰ<g:�=���=��=>��=�1==;)<��9=�Ҵ����<L�\���=1B��I�-=�J�<߁�<r���>�+<n��>A�Ֆ����o��r0>lo���K�=���=T��=��4�l�z=Z/�����^�G��	iz=�vX<�
�O
=U���X�=�!�LD�=�\�<��9=������<��<=T*=(P뽼%�_�x��g��z^>-�*>G����Z�I>d=A%�=�K��R>�����.���皽RD���=;x��K'��^=���7��gg�=��Ec= �-=����R�(�(>n�<�n��9?����������C>��	>Y�d=@�<�1D=~��A�w��3<�d���T����*�=^�:��M���ț=3� =���<d��=�n��`����eN=�[�=�٧�I������.��*㼢� =69���)����<G��;E�Y=�:��D��o�=��ѽTQ	<1�!>�=��=�����;}(X��(H=tc`�Q�<�TR=�x�<����I�,��_˼v��G?�Y�=e���\w�<�Ԑ�D��=���_=�����=�k>�Up=��L���@�3�缵�N=�ݼsٹ�ui]>d�=:=�67;a��i�;�鼽RB�<?�O��
�X҇�b��=������q�;�$L�.^����½ױ =����3s�=����m�R<ž����7�Q�$>IXӽu�W�3-
>�� ���1��l<�g����;=E��X6<5
��=�\h�}0����=O��/�C=lJ�{���כ���������%�<G��b>U��=D(Ľ�I˻���=V^]�����d���J�=>�=�ʆ<�?�;�i�=\�X��\�<�
�^��=��=[3�=�1p=k�<�9>Pk]�!���"�>���<<N=�	��7�>��?�=�������#=�C=5Ғ�E�U=�c��ĽQE<$�R���=>͚>|������=$������Mt�n�+�:���w༛X�=7ʒ��=.�K�RG�:����d罭&���<wns=�s�=0�`;���=�y�<��=o�<l������
�=��Q/��W�@F>��'���U�F���l>�!�<c+U���-=��={���!��*0=<�=|+�< 7�=n�=,����z�=��=�;����^0=��7��ӕ�<)��<���6a�=.`;� >Ϥ5�`���θ�;��]<m�̼��I���=s{�ms>4�&��5�<��a>��o�Dz=�>\�ڽ�=��w=���y�y;R.�=�c=�(1�Fﴼ��m<�Wm<�y��Z�x�����,&=i��`ˬ<|�c�K��=<P�=�W�C ����=N�$���|��;q������=���<���=D���G�<��x<k��=���=;�н�{X�2�ۼ�B�9@9�;������B=N>�3�;�2g�J*�=!�=��;%B��̽�1ֽ���y|��k�=�8U<g�N<=���X=��y<�
s�<|��D<��.�o1��	<$@ս���=�-�2�X���;�/�Z;+��� �$������<��=/A�=w�=d�I=$���&����<"J>���=٫=D6�=伽R��<ܤ�;E�E<�H4=mM�m���΀=j�a�Ԍ��ȵ>���K���6��Rd���"���>�yB�I�[=�Ǥ��b�"�f=�8f;�C�=3J��@�;,{���h��>r�<s��<Dj1<�&޼������?=:F�:7R$=���w#��d��'��=��<�ȟ�N���qU	���=�I���]���6%��	>7�S����<��������9�ٽ	
漲9ҽU2��j��ؼ1�<Տ�=�5=�Ȋ;u<���D�o��Š<'Z=�uʽ	�.=����q��$�ս\�!��e,���ռ��%��U<��J��U�}76=��������m;�n>4��:^�����)Wջ��Q����<�t<!Z��(>���Ng"�0d�=o;=���=.���M����E����ܼ�=����%��}&��C网O�̙V�~�3��_���me=���;n ��u.�d@˽�NսsI'=�!0=��ƽ��]�,5>�6��R�a�=������>;R��=����E-=�2>�c�;v�U=��=��1����$�q�u= ���G'����p�<�.��:Ҏ�w>�-�:�߽������9>"L >�<� 9<Wnټ�(���L��\�=s���g\=���<A�<RH��V�a��Ƽk ���s�:rk����fO�g�G��ׅ���=��D=��<xk�/�;��Q����&>�c�<=Ce���g�Q>��=b��gG<��~���=Iw�=b!N��㠽�����C��2�<n| >�\>3l�<�N��]��=�#�=�#�==��,>�X�;=�.���t�R8սz)����V=	ݯ����[��V��4�	���>o %��c1�O�(=/@<D��ԗ���S�<p�����A= p��7\
>n�>r]��n���������)ԑ=��<%3K��u)��b <g��;��$9B����>��`��p���߆�e��U=�x+=�1Խ}HY=���<uB����=��=s��l+�A�ͼX6P���o��C�=��޽�M�;`�㼜�߼����(�|d=���z��=o��=���=a�=�{��4dȽU</�U=-��PY�=ނ=v��=�<=�L�;6�ƼC�e=�����z٪<�W���N�`�#�D�W=:��<�<�;5�+B=������=��;=��-�>�	���zO>Ecw��1ڽ1F�<:��|@:=JK��J@l��J��o�z=kb����>��<m��=��=ܚ�=��=mw�t_E�UӖ<\�u����Z4�p/�=r"�V��< v��������;	w�=��(=$�?>�&�=��`=�ƽ���=���<����SϽ�I	�K�<|B��Z��� �=V:>��~����_s=���=���QC�<Q��=�Z�=�uA���=p�ɽRL==މ
���|=��>BJF>�[=l��C=�>�M=7M/=�>�:����<W�=�=��)=Z����<Z����� �w\�
"$��.>\>�=L��k�=��1��
�1*�<H��=!��=[i��ǣ�=-�۽�g�=F�=<i.Ͻd��<T}
<����x�<�N=��z�hw�=�B�=�Zc��R=[��;ز���;<GK��y�=���W0h=�0M=í�f�ѼDP���;����Kч�,�=M>\�⼞���1P��A!��W�����/����">HM�=P'R���?= p���e�ㅢ=��=yKG=�$��פ��b>݅�=�hL=]���\�m�g�<�퉽�T�2x�t��=���<
̫��q���U�=@77>4�󽀼>�Ȃ=-�8��Z��<{��<\��Bz�;�9=�l��B>��A}8�~H��n������S�+I�=�7�<�W�
,�<Ӷ�;|�p������o~=����I�V�_6ȼ�Ҁ�ﯽ��_=�
u�斒��|�<^=3�>�>f$���n���Lh=P�ż�J�=o�[� ���^��-�ͼŊ!����n@G�J͔=N�X�&u.>���x��=��<e�#>Kr���=���Z�5=���=�M��Q�ȼIǧ;���L� ���>�K�5Ox���=c��=�̻�
�OV����=���=m�i<'��t�T=����1,�����O=�y��A��= &=��7�@B�;4�ֽ�k<���=������=����������fzV=[ن=��j���ݼ�=�м������=���=�y&����=.�;c�=K#	>C�!>�f˽Go>?��=ـ!��&�=p�����=�??�>]w�=��,�pOw>KA<�]��B���DJ���ƽ�M��Ž=��ƽa�����<���<��9<�W�;�_�S׻t���
:�]D:gYw=���;��z���>ʾP=�m�<Lμ��|��I5=t=h�����;�Yl���V<�A`��@}=�wI=�B��?�=�ɂ=Q���r<�{�<K)�=��3���'�uX�=�0=ݕ�����<V�u�	5��#�>��=��<�6Ƚ\�~=g����\��)�=��=<���=C�>{=�q����K�1$�=�n��-5�{^X>�&=ǜg<��>�����;οa;3����Of=���<S%0=�	f=v⥼�T�&͉=敹=⟏����=�(�=i;�H�����֦;a=�=���,�8�	v=���<��<i�	AS=�k#=7��!�<��X.�;�X<�=yA�=lv���3;������ѽT���a�=4�!<�����~m��m�[36�i~�̻��<q=����9C#�)�=���<吼�)�=g�>U񕽦�м��h=0C�<2Q�=Ǆ�=��=5��=��<�>�������;��<>���=w�;=AD��QQ��]'=����,� ��=z����Խ6�e=��<6�轉ϡ='*���m��
���
=���
a�=X�/=�0=J4-==Vd���=gJ�=�=�H��(�W�򮽰y�<�q"��D��J���@����<Կ��!.�<E��=�5���9�=1��=;�Q;ݰ���B��M��"�	>Yݒ=��m����<���D-�|Ý�o=�=�g=i U�u�Y=�N�<)PĽt�=wSۼ��=���=�`��J����1�T���;�5WC>�}۽��;;�:1�봽�R<HX��/a=��3=Q7(�������:��<�Ϳ�N^�bJ�=b�=p� =���;D=)�.��pJ���ڽ�b�:�<��=>l�g=BV�=��l=�88��%����X�h݉=�B=V3�<q���ռ��d�a8#=T �0Ӽ�Pҽ�I!��s=mS<=��=(��)���k*=��G��޹=��q��Mj��덽�m�'M=�]��6��=��;�����Z=�^w=\Ն=���b���=��	 �=�����%���n8�p*<�j=�����y;��a+>������8�P�=�⪽k8I=�yn���μ��_=� <+/<��Cֻ�̃��q���k>z,�����{⺼_s=�9�6�<{�!����<�s��`=R��=>_=�N�=2(>s�=O40=��)=<�\>�I>���s��=�?t<[�>R �<	�=��>��=�M=�B�ΗU< O�����dg�=x1Z<�;B<��=���<=e;����k��o����=��&>�\=�c=]-�:j��<Ku=�ܽy	��N0���׽ޓ��GG>^Tl=�)ؼ;a����;�.��״�0�?<���=�c�=�i�,��=��i��n�n��vo=?�=� =�#=�+=�L�<3ݣ����#�2=.�=0%Ӽ�;��
G<� >�?�=�n[=0=�-�̂�=�=�
�=�p�=�J�=B�=r�>���A���>�i�;B�=��=.�$��b��$�<Wx�xԟ�+G<d���ի���<�(���o=�o�=�=sõ�{�=� =�Ў�l�-�l��=��<�`�`�s<��>K����Y�򎼼W�<���=�� >o�;>�];���=���<�g��؄�=p	~="^/���=��.��g�~�����<�>Z�8�;��=��Ż0�<=��^�q�=����*����o��o�׻�:�=�U�=ێ�=3�:��]��=����2��o��=�kϼ~\q�B9�5���a��6�==\�=���Y�庈�D��&��'O�؁k<>�<���<�8>>��
���<�Q�þ�=��;�b<��]�B"��Hz=e�7=�2�=���<�]=u}=�}�=02=WR�=�Z���g��<9�.��R�����=.�
>&�$��L<�F@=���=��=P��<�e�=f�d����@�bE��d���3�=�=K2>�g�Ri=@=���=�쉻�~=*�Ľ�˯��t��H"V=@��=㤘=Em�=���<\)�=;ҽR��<��ѽX�o=��#���a<��˽��=4��=����oٗ���};{=Ž;�ǖ��`��x���P�<��j��Ȟ� ͔���W<�6����=�zν�a�=��=�����f+<����1�=0	�<0ż���="�#�U���ځD��,��H���G�=�ݼHm�L"�<d����0�<�#>���P�5=�1��o=�<,��� �a�A=z޺=Xu�������=�1x=�*=�={vѽ<BQ�o!�=����=XF�����4��o ]=��꽤*?=�>^���Dμ�o�Q3�tN�<	����O���B��)Q����X�=�@=ߠ�=X�x�C���SR�<�%<���<*d�=�<��>2�'�mܼ��=����)�b1�=p.R;Ksɽ��5��,+�� ��FDj=���5=T����O!�@�=s�X��Gd�t�=z=���=��=�(S>�< 輕�+=u�<Ƥ��@Ž��9o��Ґ��&>�]�=X$ڽ��ӽ "n<�D�<�-��У=(Žiq�=�׽�\>q4;\�#���]='e<sX���#�r�>�R������lt$=6���k�i�J�(�U�]��Z{<��K�/��<�i>�~��]�=�_�=��=�Ws�o��Pӳ��29=߫n<M�������(B�W�
����=|���[�-(9�w����+��E>N����=*w��ʤ�v��<��=p>�ɻ;���<��=(X;�ş��N��<㢾=�����=d�=��i�j[ֽ�$�=`��=П=��<�=�O����<�]��=�p<`��<\Y&=9z)<X;���9=
;�5�;?^&>��=�(=�֑=��<Jn����弒��=��=����[�5��=n1	>�����<$�4���9�0��<q�=�!�=z�=
5üP�;#6<�;k8����=��=A-=?==��<0"�������]�=�&=���N��= -'<bX���S1��B�=�;�\��<��߽HE<���=pEO��1���r>�*)>�mB=|�=h_1���>�QN��Aֽ��F�Y=ŵ������*����|T��s��rX=pC�<�>��=��>Q�=����������<�����4�=�y�=A�彇ķ��6���=���<��=�'ػ�# >�5�=�T>�9����ʽ:�=�c2���r�*ץ=q���D׽���sʡ=��$��"6��,�=�{�<�9��,��of ��Wv�=ü�x����Hu=�:�Kp�ht��p������I#�=o2z��O����#Z����q����/�="��<t�#=T�t<����@��ɢ�=ޛ�=�y����$>��P�[��O�g��E�=���%�Ľ��<���=�J��#�=�9n=x{=w����:>�4�<v|�4g���:�X��t�,>����*2�<�ɵ<Y�������{��- ��8��=��W��Z.=�#�<�G��)�!�����j�8�T<��!>��p�<���=��?��o�2轾ټ=��n=[E�=Z���&*��䴽�~�=N����=�@]<�*�mZ�=�K��4�=�wG����=[��=k0+���->Z¯�-���b��<p��=����:9��ꑽ�u˼L�%����� Y=O+��� �=$ ���?T=*��"�
����ٝ�<�'��\l�<��	���=8��� �=�~a<���<��A�1�ɼ	��v׼
�>���=ʦ<)��<�q�=_�"=>$���<�"�h�)<�Z4=A�o���.-�Ư_=�p=��=fSg="'���=�>��8�mY���l=��*�m{�=�>
���۽dh��,�d������>c�a=�Q=�����*��=�ڨ�?����=/r�<(+=<e�\n>]��=2��=���<�〽�v�= ���օC�Zڽ�\G�N�Z�<��(=��:�F�I�M�>n�ļaY=hY=B-�sm�=Ѧ���#�=i�j����= C>.`�+4_�b�
k�=BD��j*��R��Vev=y�E=u��<R����䛽v'�<��V=k�=��R=���;-�����:�%=>6�=���z=���=����y;��a�i�ȼn��=���=h_$<zQ�ra�<J��<��T=�l�<1�=&�(�/�>�,����M�T�'���G>�p�Ή�<��%=�ϫ=l= ���m4�=^��
d�;u�+>�G�;���=]v��U�=������<���=tI��}8�a�{�bj=�Z�=�K�=�hr�MqS��&>�Ht=����gdY�{{>�ST=�m�=��=})y��`J=����f�q�v������<�����5��)�=x�T�Y�
=��>H}E�^�t�O�
>�䣽���'�ӽÔ���]<�p���B>CU���]m���i�0����i�f������0^=7G=�2�<�����45�(D������=�7<��F��=!
<!���p�ߕɽkkм7m޽�=~!;FwW�v=��f�=a�r��<���G�ѽ��<�nG=�M>��]=O!���<
x���^���C>dP��#��</.v�[ȸ����=����:�<����g�Z�W='�ὡn7>�ٽbӼ���<?`�=-FF<��߼��=�|�腵=��o�/��=���&�=k����}=�?����Y��O������νۅ������m��<�Z2���\{�=�� ���=f��<�0�=Wr+=2�q=�A�8��=(�|�mI
>z^X=�1���U��د��߼4��<K�=�^���!�<�5���<E�={����F4��.��t�v:#Ib=�3�=�~�Β=���=y�Ҽ��J;��&;���=�rM=���=�[�:I�S��w�=Nf��,x��,=2W�Z ]=��h=�&��x!��Dh���Ȼ�	���H=���=񔂾:�>� �=a1i���4�]F�=]��=���9���7�.��%�?w =��=*��3���� ��F��a����B����w�j��A>#�V�����п=�y���ٽR�x=�&_�~���϶�:�2�=ȷ��称=�bt<O$?��k��Bq=����=d,�=E��=Å������.��fǁ���ʺaK���|/<��½�$�
�����+Ż�%����g�����&�,>��+=�,��K�B;�d"�sYz������#�=��!=�E`<s���i1��)0�"�Q<�A<r"
>�Ӽ$uP<���=���<+G�}p�=f.Q��><��<	�bb�;���f�;=ڋ��?�=Α#>޽>��=��=��:�0��;�����8���<�h>��=��N;a:��BQ�<���-���/�H;`{����=Zۅ={���4=�$���Ǻ�"/�z�s<@Jݽ �Ͻ��+O�=-���4�=EG=T	�:�;?t�l޶<V�=�4�QH�QrƽI9����E������>��=�<��I��A7�������4;�C �2½>�=�A�=cDB=��ʽ�i=��1=�&=�jX�2�3<�'�����=������< 3m�դ޽*��=��Ek�=���=�-��J�ui8=�=�t.>��D���{����=��Ž�(�<����(�<8	��p�*�k=�m�1K�b�>د}=@`�<U.:�Uz=ɨW�4������=����Ͻ���`e�M�%��䅼�' >Hx�<�U�(1��A��;�����o>��q�=հ6=$ʽ�Qz=����_=�e>'�=���`�<ڠ���������)�\���0?�����	>��p���=�=UC=yOX<.���"��n�m�욌=�����v��T=0�����:�|<�����-����p�������$�c1=��=�=�W=4��_��h��0�_�mۢ��)=10�=]��=�<�<~<�=���=���*�D�1�5��1���	ʽo��6��<�[=�Ff=W����?=D=os>���<Xq������=3P<�
�=�&���.>�!f=���=���<vh>R+>�"J;24���y;=(�I��<\/f<�[�b, �!i>�� =	�=H��a̶��Ű<��*��\�<a�Ӽ7��D��Y����>Kvٽ�v���=�,�=+���2����,���e�B�a��-�&�9��]�=�c��.�=wv�/�<腚=� �&�=��I=�)�=ͪ>!.%>RD?��+��OV��<��<�l����=Q !=(��XX�;fY=M�:_���k��c�=�j>>^<�m�j�td\��$�<m�^�F���j������n=��+���ż�	Ž��&�Y�.=+��= ��<?m��h���	V >ӛb�k�D<c8<<��k��ï;�"=����O+E=Zi=B�A=E�=�/e�HO�59�<g��Q9��w�;�@�=��C���=G�7���Ľ�
��`�Y�H�<�=˼���8�>f�н����J��US�_�=��<af	�,�߽=�ɉ9���=m%>�kҺK�U�޶T�zv=��<�ټ�a��] �Ϋ�=
@��FL�=6bk=$�=�,>+��|�g=s��;<�=��=tlǼϼol�=p�����<�$=Y;�i�@��=��=D��=�=<9=b�4�ò���=8iJ=�S�<4lɽ}}�����=3�;{���}g���K�<#��;K��;;�g����=\[@�����A�=�2o��E�;����U�=�t�D�o�^��έ�<i���ǎ�8���yr�=�m��y=�"X>�u�=�x�<wc�����;0�<��%��x�;���=�d���=�)�2}K�`?"��#һf����<$�߽O@=�����=�X�<�X��H!�0��?�L��&>!�<�<_=�޺<��<D��<:N�=1j�ʹ�=�	潽�f=v�>�P6�����tV��F�=_��5�=�0�=�;���o���4�=��<��U�����{�B���]=܍'>0� =pH\<� �=���hyӼ�N=�	�����=�K��,HT=;����
�N����tc��b<��2=�K�<*�.<{U3=TX�p�;Qy�=h�8F��=�<���=��罦J�V߽�@�<;Z�=�j>�s���+h���׺k�	>W_	>鑈=cħ<+�p<�s</.�=炆�L�i��O� ��=^6��ܤ:��t<��<�r=%{:��ۻ4�V��P,��;F�1=��j=R(����?���Y�b�b�<lˣ=������=߲,>~n��.Y�� ="�W���=-x#=J�=ߊ=��^��>��ƽ��{� ?���=��O�So㼄���������p�=��2��	=z�Ľ�6<b��I@��6e��7zŽ$$+��1��o{V=��ܽ��{=���;'�=�9=E�{��g$=��7<�mѽ���<�F�Ʃ����-;���= ��xƇ=��P=�=�p;=��<|L�=4@���S�=ϵS��"<���X�\=U������; ��=�-ҽƢ��m��=�"���*n=������=F�K�#��.�D=ԂX=pz/�p�����=��E�Q��N�=�罝�>3t�]��<)k�=s%�=��J=�!7�0�=��-��">�,�>N�:���=�QV=�ؼS�;�B��F<���Q=z	̽6�=�=�=
e>�j:�����n=�H�=�=(=��8�O=�=�=I)�<�=S�$����b=��ǽkI]>>��E�<����O�N�'
29g�7��%>w������Ζ=�}��h=�>dj�^9>,I&=P�=�I��ܟ�=�,�=���=@ҽ���+���ǜ=��=XV�9���ӷ�=o
v��mm���o=� [=��+>J(��?��<%�=̨=��==7�<1M���\�~�;��>F�	=I3\=�x=:���u��=����t6�=�M=1�l=�Ġ��8�=Z�N�s���MG=:c-��L4=�,;O,(=p��=c��=�e=��<��&<\���<_E����=�o��RP�=����T����=܃�;��]���P=f-�>����:>@�;;�����=�%�<�NS=,E=U?�=����W=m�=Q��=u��x��T�o$=��7m���~�$&{�����Y<}���Ck�=��N�μ��=��>z�<��C=���=����{���Mؽ�t�=������o
��dm�>�>r�=55w��m�=��1=�h/�BĊ=������=@���q���нg����<�8>�p�=4)�dX�����<&ݲ�(�=�{�Jg����<�Ϥ=�	>3�;�5�>=�����~�D��<6��;�չ��=���<�ؽ"�!=&�=0��=
߉�d?�<r�&=).>������=p�ҽW�C>�를�	"=fQ�=,�=l����=��=O�%<O�����<���;Yb�`������������`�H1:R���������؂>Ji�;&
�<h��<7�<�+T�]II=J�)���ٽ*��=�{������C>&Z�=UX;�X��㈽u6����=�mL<[Y=�v����"=�������=�.l=�k߼M ��^=��>",����k�y��jg�og=Q@���
=�e���<��2>'���X���x�����^�p�(�U=ㆢ=�C[=���^}:�FԄ=��>A{<���;��(�V�;�;K>��<��T=�!==�	=krQ=���;UR.�K�=�<��	�I�9<7�ӽ]��=���p47�t�F�l	�����N����=��{��t�=�#<��x��+���&=����^��<�1��\@�<�����;�n����Z>v=��
�����?�
>�5=W�;��=}��;><��Y%�=�3�<Ƽ>��E=n~ӽ�R�����*+��j:�/������9��=�0�:�Ⱥ�K�Ľq_
=�Я��i���v=����3`�=�3������d\��`�=�E:`�=A�=@|H=�o=���h���1�=�^5<�l�=��r<���������<��%=X��/�=^�<=��};q&|���Խ�����=':��4_�>i��YB�<ƈ=h����=.⊽��K��f���]�<���<K�=.�!:��?��=*8=�z=�zڼ���<�#E����<(���D���iѓ<�)�<Fx������4�
����=۰
���|=�u�;6Dƽfe�=)�.=}�d�{i�<@D���0ɽ����C�ν��#��=�O'=jU[=�+���_����d�w�=Zy�<�)�=W�G�	H�=��Ͻ�=t����}���f��3�6�t�=YA�=�t=��̼62>}>���=/�$>�8�s4=����Ž��^�O�y�v3x=�'N=J<��.Tͽ�p�=��=q<!�=Z�=��>��X� ꤽ��^=U�">�=O�=��&���q=�<���=���=j0>�=QG�<���.�̼u��=��Ǽ��8��P��B�;=�ջ��_<��>����WR�~�/�lh˽C�%<|�����^�:�#�=E��<�e�<GT�=���?M�=!:^<���<�c���d>�� =i\w�MDr=�"��Ӎ��=޽�=�4.�����5�=�jܽ�"^=��=���=����tO�=�4=�
{����=����o�<`��=��ӽH�v�_ep�.�>a�ûf@����H<�%K��l�<�Ax=�D}�8��9=��4��F���9:>��=X�=���R:=���^ϼ��:�y�����껍彵��.n=�߬=]��<w�=���(>99ν'��;�B�<F:���(��z><�Ĕ=�!��f��^="+�u�~��TB�,����G=Zs�<���2;�BG�����L<}�=�R�=�I;;�:�=<o���t=='~���y�=#s�l�<$N��J�<�l�� �c=�0�z������ļ>�=b�J>v�>�	��1X����=�Ý�_��<��=Г�<<��="��=1*�<&����<�K.=�8ܼ���=�Y�=k�2<���|j<?.½M�����<����ۄ<�ɓ����s�
���¼�A�<�ɩ�`���^B���B���2=��>��μ�F��g�=ͼ�<=�ĥ=J��ڽy��>.<=�j9�=�R�;o���ڼ_�p<~��V<p��=u���"��(2=	=�w���/���ڼ΂�n�6;Pz��$���ż��=��=%�a��=q�Ȼ�o��dN����̹����p���>��m=!��ڢ>���� *�e�=|<:=��=��=��V<�.M=cH^=�����8�ܘ�=����|��,�<A��T���N�,=����ݎ�<G����A��<
�:Rn��O?�uO�;2�=�2*��J��
��=BЭ�.׮<G�2=m��<�B��$O�<EBT<����
�k�d�r�E=�<�<GY`<�C�<�0=g�f�0F<���=ǚ����=W�����a={�<lj�P�r��:ǽ��T<_T�����i�=꙰�V=�=�ا=�mf��?]=�H�=S���X��{A�<i�=���<;u�l�&�
ꪽ�=<)�=_��Ff�=�AֽJ��=,xp��ʐ�q����#�f<�����x�=���S�=7	v�j�q=��H�i<�GG���������?�༫�S<)u=!o[<^\����=�Խ�f=%���y=�.<��=L��=������^k�=���_��Aк��Or=Z�߽l�߽�<Ti\>�'���6�;�)E����k}P<�r
=� ��eꍽ���8�<�h����<-&,;�=ϭ<+�=FR�����
�A=��=�V�J�c<vP�=��=����;��m=�g��p}1����tM��xp=pc�L���G�B���렼u�#>3�>}��:0,>�z�=Ȕ�I"̼I�~�lA >q�Y=���r�;��y��Y>��O���C=�j�<|��=t���$���$<D��=��=��썽�Y�Q.>{��	��=���6����<��D��	��t�H=K�W�����J�ĽmSI��]=;f	<��=
ⱽ9�<�kD>E,;W��U:~�T=^�-<�8�|@���	=l��=���=]�=]�	=�E��JA�=3u=Ә���-���1�=��<E��<�RƽH��=*��=�2½ӡ�EB+�f�X�e)��c<�	��.�<�׽R�W���q����=��;b��=��R�.�B<��R<�M< #�<� >�$`=�,=J�=��6�4�;)Ğ<�fؼ@=`�Z�d=�  ����=C<���=���=��\=�l�Q���0u��ý�J�gA�=m㉾�R�<�����g���6����$˽�v[�_��&�L��Tk=4����q=�%ۼ�ɽ���<z��I�ͻ����=�<�l:=w��<�'̼�Nz=�[1<d��=�qü .������Ʋ�ސ����=�k|=vϊ�0ת=��=�F<�!=��}<1���#�t��~7�A7�<�?�<G8�!�9=ذ��=<������d����f`�
��=dWR������=�O�=uMN�������=�1�)k���9y�0�S�jo�=�뾽X����<��߽�-��-�>��u=�-k="̈���=	�> ���l֚<��v=��=�±:U)Ǽ�+�<�/<�7qķ^r˼��Q=ρ��w->P�=����h'=HV=�;=��ýq�<lػ�m��i=������=�6Z<ir;�A=�k>X�O=���;���G:y���ļ �������p^���:�⾼|?��?pe=�ܿ=I&W���=�=
�pJ3�v��=��>�(A���=ǋ���Ͻ�I����G���?�g>��>�@㽛Ƭ�V� �|w4�v$W�<�\<U�=I_�=��<�&5=�P �e����Ը���=_=�J�=��#��r=ʽ�=�%�h������ֽ���=!�����8=t(Z<r�5>��<��������^��K�<z@=�M=�}G=l[�=$�ʽ}���)���=��=�ʿ��"�=�0Ͻ�u���L=s�J=�p���t�=��e=:>[3 >s���MA�g�[��K�=i���湽6v5=�0;kI�=	J=�v�;��;��<��T>_>��ICϽ�x=,�U�N�(=�̄��*�<+콭�׽����$8��"�����=9/�����,Z=5��=u|�<�E����M�l�M��+�N��K�Qڄ=Y_ͼŻ�=�[�=��=j$<x-�:?�=��һ2���Ǿ�+h�����[�A�z��<�ڽ�6=p�'=J��<� ;`��6�	���4�`��ŝ&��*��� ��笽���$�:���I��]4>[�=1�=wG>�-�Bܢ�] �=܈�=NĽ�L ��L�*r;��&����<�=J��۽Hh�u T=6��F�=Z=Ev	�Z%O=��]=Τ=>���Ya"=����_�<��ӽ-;ߥ�=�(����,<�>|�>�1ҼY$����O�&�<�@��TC�=fG�=u�=��l=fF��e��]��;���<�����_���=�r��⢽�(<:r</�ż��J�<P��=�4�R��=>$�ZҼ8�i<���쵫��(��<ۂ�� +<6}�<:��;��=�d(=����E<g�ܪ�=բ	��˽���!޽�T��Z^�=�R=P>���f=s�f���>��=^4�=
<��<�h���{���/�<*s`=�08�Y��O���n���꼺��=�%��
��y=E�=b��������-��0��s�#׽��
F��YМ��d�<Y�=�Ľ��
<���:�^'�|B��/����<܀��o���\�=�N�=�p��7��:iT��oP�d������.�=�={9�=�e=#]��9`�<M�q�����l���=*G���}@=1¼���G�L�=�=���cռU��5A�g�ǽ�O����s8><��N���[���9=��|����<m�;$���g#�*D^>����H����=�lH�.�Ҽ�8˼�0�u��=�k�<���v��<�����q>��P��M�R�ռ�`���=Y|�<X�=���=��=�oh�.�@<���=.b�Ol�=Y*�<j_�=����e-����=󁄼�,X=hp=5!/���1=���= �!�%-J=�=~�*;�H�=���hzB=3�?�
㚽�L	>sԼ�#ļ��>��<�x���=tfu�-�R=mb��g<��C�tm�<�+�=W�𽑖�� ^ּc޽z�=A̺�E.9���Y���x�.
��mK½�=���������1�=�N>'y�h>p��"�b=z��;S>�&����k��<H<v�'�]'�=���!i�=��*=ܨ=iiU��`�=]z=&�=?-�=��n=pÃ�#�b�`L]=�o�E��p��,�<�F�!�
�Jy����==�`;��=e��u���r�<��=x�1=����1V=��>a"!>����?G����X�����
J<�=���t��=MS�;]!�=>��Z��M����=�(��|R�Y�A�g@<�7��re�=��S=��:9�<f�=� [='��=$�=#6t��j��`�ڼr)����� P�M���ͻ��������Խ}����{<Ȁ��9�<����[��gL�(>:��~텼�Z���"��g��/�V�n��=�j=�9G�v�>�,�9�Z>�B��D�8>i�X�J��Ԉ��ȼ��½:�=N�\�4�K�w)��%��΄�2=*z�����䨽���VS=�jx�vF)=󑴽���=0o�<��=�/C�x���BE=^[�=~RM=�7��>�=�7>p!�=�_����=�h�φ�����:�|=N�;|��=a��sF�ub��x�=�oмm�ͼ��{=K�&�V�S<t��=]=�p�����f�=����o=���M�!�0�&=��нb@>���̅&��/��(���84==>q�r�f��<�	=I�����������=Z>��S=~f�B��;�?���j��d/x����'�踄�ipѺ�
@=��^�M�=��b�F���S	j<bW>{��<��<�r;]w�����+�(�Z��=l���	�=�K�<���OO��d��;��o��ɡ=n����ϽQ����=��޼d
������W�#�\���B�<�;>���<|H��~��1�;=SV����<&>��Լ�@���K����o�)��=�;<=8̽�"	�N<=��/�mO��ȍ<�ˑ�|�c<�z����>$y�;]���Q���"�<	Uܼ��4>J��=f��=!����.�=�'�=�n8,��>�-�=�O��n)�EY�=\n�=�=�<w��0�0=����Y��L�,=��=�>����������Z=Yϼ���<X!�;�&�<&н��0=�#O=N��=��+=�w�=��V=���=0�$�c���Ҿ<�.��T�=�h�<��>w������/�߽kF�<C+�=΃&=��>��.���%7;�H����;��νcZ�<xQ$=��@��aI�����?x�߹#=�Uf=�U=�km�2;?��_U�+�<��K���E;��4��Rǻ�BQ>��/;�&>����+y;*���[y��c��=U���ު� �s�.΅<�F���K{���C<�����Xͼͥ
=fB��N��[잽�
����j�X����>29���<,Fk����=S��#����=�a���^3= �.>[��<���=�ƭ��;�S�Z=���=�@r����=+�i������=���=AƢ=�Ѕ�a��=�~~=<�O={������/�k��<�0�=�#s�&�;�A>P1�<;ﺽ�!��P����>�<�o8<�齈�˻��e=�d�<Y4����:>�s�����.
��<��=^��=�D%��fS�8��<w==�x"=ڏh<�!�9
��;��=77J>e�	=�h=d4�B��=�Y^=f����սV�v�y佱�N<���eΒ=:��<ayI>$�>=��=�	���ͽ�䨼P͞>h��==V=��0���4����-RO�wsj��m+�r4�=0,~=5�!= �=U�;gU�<SD������N�[���<*�=��^�J��l	���L�|�B��j��<X�K=���C�w=W��<d��=������<3�=��<F�w�`ݻ2��=��ǋ�9�=����N(�<7���{J�oE*���}���M�}�e��C�=M��<{�=/�=����E���t<FR��e���������ؽ(Е=��;�6���=#�Q=�G��ʼC=��=�d�Qȼ����󊉼���mS��_=:�)"�=�:����C"&<5Q@�4�޽R�@=~]��7��=��*>޾g�w�����<�]˽���<�p���4 =�>��/<.V�=7�^=Ŭ��-O=���؊�=������� ��F�9��Ú�\�*�V���&�H Խ�;(Cw��| =rƷ�$���ȅ��w8���>��X�i,:=��o����=-&�=��?�W��T�0�O���O=>X�����p�!=	`3��U=���t"=J����ȼZg.�VoZ��0>�W=��<��=�I�=58Y���9��ﻓI=W�'��^��ܥ��T�3;���=���=�yh�5���K��=)n=�3����=�U<.���Mq�<0�L�[���q�=T��=���]�;�U=�K�="�>A����������1����7�6�6=��f�I�¼j�u=�p����=(w�j�н̽�x\=��=O�=oyc�X��=[����=����;��)�׼�=�m��թ=��}<!��_4��>��"�lɇ�� =�����:�rI�<~+���ĽQ�ؽ����A���胼�#=K2[��≽L����:��E�=DgR=gkĻ�=� ;�4=��۽u�����<C7n<GjG�Uj�<S��M���$b�=�л�����_�5p�����=�z	��l;u�����=�n=�Cw=�"ٽ��=��F=M��<��a=f�&�D(�=5�ƽ���<�'�=���=��-=_��==�>���g >f�Q�@���U�;}%H=�� �n;	�����@�=�P��}�|����B�ڮ=���=�\=�p�
��<N�ֽ,ڟ��=��9=�o?<Ad��Ǚ��Ǽ8��?�=��=��=@4�=�4�=-K>@��P��"+
�����/=�Jf�������=&a���!�hm���7=)�/=��6�b��<�����?=E�l��F�=�>�<�`�=+ZP<BC�=qF��Ww��� ü���=�\@=?Uh<�>����5���;�=��F�L�6�{6��s[�K�]��$�3�C�5s�=F�����=���;�1��ݠ=��&<��ɽ�(>��B���n��"�=S]�=��u��� d�<~D$�;�9�2���4�x<�B��߃>z1��#=�#>�o>G�����]H:���b�8{�=�:�=�-=q�=gS�z�	=:(>�,�Z�q��T=�Z>��c�Y?��н~3�0�ϼ<o;a�H��Ž�MR=;ýB�ż�75=~Q=���*�=~#��s5�=���=�q-=���zV4��4D��w<���o�=7/�<�� =�k-=�<!��>��;��=�X��t9��j;<�t켏�(>����6H��=�s��="K=�3���������<r�Q=��<�?9���=b�w��m �gY�����_����9?=���������8O=���=�(E��M2��6>�;�=��$���8����΁�����:��=r�=��U=���	>�=P=����,��#�{�����k75�ͧ-��,=֢ͽ�)e=4���@��;��u=�/��� ����*>�������<Հ��:3��ńU��w��2�=��X=<Hq=�<�<��NJ� o�X;����6�=:9�<\.6�t�j�7=��=�[�<n��;�w���=Jr&=C�>��KT�אK=w��=;>9��<'@��;�=��3�`B��2���H�O<��>E���U=��ջ�����	�2O�<��=��<�)���c��������=?$>0>c�=�<�	C=��޽5Fm�Æ=I�=����Q�ü��<PH5=�^<�0=3��=��8����������(��=�E�=3���?H=��x�㘷<�t>�T=�K�eVs�<�>é�<�U51�>�fv�� ���Ik�m8�=�f-���A=3D=> a�=vK��sʹ��V�=x�=�3h���ᾠ=��ͼ�I�="VD�k�s=*S
=�8>����x���I���u��0G�<�A]�϶ <ʈ���=P�;�OټZ�=}�J=�\5��HS�d�>=��=R2�<�Ja=�����0�Խm��D�(<�oǽ5-�<�"=��5=5�<�ީ=�Q{���m��6=J% ��d�=}����C)>��M�
�2�)[�<Ӂ =�W�o����?-<����g=��=}�.�[m�#i(�c	[=Z?��m�㽧�ü෯=Q%�=zȼ���=6�����B<�?<jHF>�S�2�=�NR<��>����3-�[�	��oH�.����v���=��~�'@��/	��bнB�_=��=q�=�=�=.�½W�P=h8�=���<5G�	�`=���=0t��D�=�ϼ���ƽ$=J�>URK�ސ�<<���=���?��<�N��q�E]G�ڭܽyj5=�[��z�=m�<
Z<m}`<�<7�7��d���=墍�z������=�a����=���=Y��=D@��8�ɠY=��k=�8��D��<Z��5E�<�z�<�c+��<R=	� �Q(�n�	�m<�a!<�]����>/��\��Qe����6���p=&=��̹����<=P5Ǽ�*=٪>B�=���=����L���3�=?^�.����ýV)�)\ʽ|����<*2�:�_)�d��=� I�)X�$�����]��k���<�=���=]�R<$R=�����?�N�=\��7�E���l|*�JƼ����=�������<���=U/X=����� >ϰ���>���}f����=3�ռ|p��4���{ �!��&�&�
/�<):\<'�=�=e=]`*������:9<��>���=�1�5�3����� �;�r=����'=��
=w$z��@=�#�����c<�;���G>������=��?>�=+������=F�޼�<> ����5=~z��.M�?�=��<$�'=܄� �=�7u=�P@<�ˍ��W��f�S�.>��S=�{�<�$ �:)����#���	<�ź�G=)!�]=/f��T�o=w�=�9���&0=�dU=+)�=��J���lE��փT����3<=�>��W>��罄x��b<�=�>���{���Q=���$�)�/����<4_�=Y2ώ�.���U=�.���>�Ky��i߽ݦi�lA=�0>cV*>��JcM=�
{=��>C�L=����ȼ���<<�<qJ�;�6�=�m=�8J�9����=w���Z�=�9)=/������+��̙�=�tz=*� ��>��<�	��܊�4��=���=m�߽�#�=�P=O� >yt��ꖼ.u�=�Qq=��"=`PU=Z��o�����(�=�g�q�%=M�b=���x�z�f|�<����#>=�ӌ=��ܽ��F��a?=+�������̼ؒWVv=��;W�%�!�=$r;]ӈ���X����=P͈=�
9��s=>�ν&�"><=;ӝ�`��=?>�<�0�=r��@��&"�Zg�<㞽ؚ=4�g;2�=��>��=	%�<c4���-n>%$B���D��D�9u:`=yoB=*p�K)��s3��d�=��Ͻ;̋=��w=��=�JL�!*>x��<*���%=N���=�h�<V�=z0`=Nǣ�����;H�<^┼���Л9=v_�����_�;��������Wq�,gӼ��=�^����I�(�*>K��=_����Y�-ϕ9o�E=�ٽl=/��:��=U�9���>ؚ����⼈&]=m�w��:>qp���J=�䎽�}���=(;�6����<݄�;%�89���<M�y͕<�n7�� =�.\�X��;� =�fD���D�w%K�O��=�赽ɸ�	-��7*� �K��s�-dѽϭ$=|�=	Ľ����Խ���~=�P����*�8��=t� ��Q=���<�S��h{𽑻	��v5����=C�۽"�&=W�J��;H=<�;�8��=Cv>���=����n=k�����=y�"�E�=�|���d4�è=FA=/�>O_=�eǽU7�=��> �=����.O=^�<�[U�.��=��>�X�6�e<�L�:G�=7��������̚!=��*>� 3���<�<_���e���=���=�/�<>i��</���
ۼP %=Rָ���8�'�&= ŏ�k�=���=.>����=a�м�SZ<�(=�L����μ/�<���<ߐ'<=�#>���<�=�(���+�J���9�<$��(��<]o!>	�M�/%��@D=#3�=Rh0=�)=W`�;b�>ì��]������<ؽ�v3>>\=P�=�༹���CΙ=������[�}������=�)#�?<'>��Cz�������]����=����H=@��=���<sѬ�I��
)��|w<ף=�˼}6�=a	��r�= Ľ����=�Z���<��w��T�=<5�="��<5-!=(�A��=6�=d:�a`7>�z�=�>#XJ<�ā��@�-k�=蟼�(���lP�ǸH=��<�5�=���38|=��=`_ʽ5��	->|=�Q=��=��:�,Q�<�rK��`�=S�O=��=��F��Qɽ��3���,�9�=��<Y�������n�=↔�_4���5�x޸=�vý|�=��׼��ؽ$Q�U"���G =#&�=L��<�e!�_����
��Tf��:^=���=ՠ��t�<�����c��A=�\o���=��=<ҟ<"�Q=�&�<Ld��*�\����=U$�=ȼk9��S>��=4�;�u<"Fb�#�=�Bϻ�$f��s�=v�����<ߕ����1=�o�����\�=)q2���%.Ż��ü���<�9��&½">�U��o�c=,�L=��-=��=8>��D<�̾�|h=%�y=<��<���=}�=�F�=��=P �З��Ĺ�HQ��V��<x#�,=�@I=��=e�Ӽ4���mڽ���/�����=g��h@:�
މ=h����*=r�-��.>�ٽ3R_���#��>��;��]=�66�z_�<���=�Rd���=g��;�]>y3��5�;�᥽��<�V��~D���=�s�<~ ��ua�=/�=�O�=�`~:��4�p����=�-=_���=�(���~���֮=�7=K�佒6{�C�
�o1۽�h��[]��M>08�=H^>Dt�E��<�$@=�ڼ}㱽� >+V�=�/>*
_output_shapes
:	(�
�
lstm_1/recurrent_kernel
VariableV2*
dtype0*
shape:	(�*
shared_name *
	container *
_output_shapes
:	(�
�
lstm_1/recurrent_kernel/AssignAssignlstm_1/recurrent_kernel%lstm_1/recurrent_kernel/initial_value*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes
:	(�
�
lstm_1/recurrent_kernel/readIdentitylstm_1/recurrent_kernel**
_class 
loc:@lstm_1/recurrent_kernel*
T0*
_output_shapes
:	(�
]
lstm_1/Const_1Const*
dtype0*
valueB�*    *
_output_shapes	
:�
y
lstm_1/bias
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
lstm_1/bias/AssignAssignlstm_1/biaslstm_1/Const_1*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes	
:�
o
lstm_1/bias/readIdentitylstm_1/bias*
_class
loc:@lstm_1/bias*
T0*
_output_shapes	
:�
]
lstm_1/PlaceholderPlaceholder*
dtype0*
shape:�*
_output_shapes	
:�
�
lstm_1/AssignAssignlstm_1/biaslstm_1/Placeholder*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking( *
T0*
_output_shapes	
:�
`
lstm_1/initNoOp^lstm_1/kernel/Assign^lstm_1/recurrent_kernel/Assign^lstm_1/bias/Assign
k
lstm_1/strided_slice/stackConst*
dtype0*
valueB"        *
_output_shapes
:
m
lstm_1/strided_slice/stack_1Const*
dtype0*
valueB"    (   *
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

:(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_1/stackConst*
dtype0*
valueB"    (   *
_output_shapes
:
o
lstm_1/strided_slice_1/stack_1Const*
dtype0*
valueB"    P   *
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

:(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_2/stackConst*
dtype0*
valueB"    P   *
_output_shapes
:
o
lstm_1/strided_slice_2/stack_1Const*
dtype0*
valueB"    x   *
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

:(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_3/stackConst*
dtype0*
valueB"    x   *
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

:(*

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
valueB"    (   *
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

:((*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_5/stackConst*
dtype0*
valueB"    (   *
_output_shapes
:
o
lstm_1/strided_slice_5/stack_1Const*
dtype0*
valueB"    P   *
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

:((*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_6/stackConst*
dtype0*
valueB"    P   *
_output_shapes
:
o
lstm_1/strided_slice_6/stack_1Const*
dtype0*
valueB"    x   *
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

:((*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
m
lstm_1/strided_slice_7/stackConst*
dtype0*
valueB"    x   *
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

:((*

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
valueB:(*
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
:(*

begin_mask*
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
f
lstm_1/strided_slice_9/stackConst*
dtype0*
valueB:(*
_output_shapes
:
h
lstm_1/strided_slice_9/stack_1Const*
dtype0*
valueB:P*
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
:(*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_10/stackConst*
dtype0*
valueB:P*
_output_shapes
:
i
lstm_1/strided_slice_10/stack_1Const*
dtype0*
valueB:x*
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
:(*

begin_mask *
ellipsis_mask *
end_mask *
T0*
shrink_axis_mask 
g
lstm_1/strided_slice_11/stackConst*
dtype0*
valueB:x*
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
:(*

begin_mask *
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
b
lstm_1/zeros_like	ZerosLikelstm_1_input*
T0*+
_output_shapes
:���������
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
valueB"   (   *
_output_shapes
:
�
lstm_1/TileTilelstm_1/ExpandDimslstm_1/Tile/multiples*

Tmultiples0*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/BiasAddBiasAddlstm_1/MatMullstm_1/strided_slice_8*
data_formatNHWC*
T0*'
_output_shapes
:���������(
a
lstm_1/stackConst*
dtype0*!
valueB"����   (   *
_output_shapes
:
}
lstm_1/Reshape_1Reshapelstm_1/BiasAddlstm_1/stack*
Tshape0*
T0*+
_output_shapes
:���������(
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
:���������(
�
lstm_1/BiasAdd_1BiasAddlstm_1/MatMul_1lstm_1/strided_slice_9*
data_formatNHWC*
T0*'
_output_shapes
:���������(
c
lstm_1/stack_1Const*
dtype0*!
valueB"����   (   *
_output_shapes
:
�
lstm_1/Reshape_3Reshapelstm_1/BiasAdd_1lstm_1/stack_1*
Tshape0*
T0*+
_output_shapes
:���������(
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
:���������(
�
lstm_1/BiasAdd_2BiasAddlstm_1/MatMul_2lstm_1/strided_slice_10*
data_formatNHWC*
T0*'
_output_shapes
:���������(
c
lstm_1/stack_2Const*
dtype0*!
valueB"����   (   *
_output_shapes
:
�
lstm_1/Reshape_5Reshapelstm_1/BiasAdd_2lstm_1/stack_2*
Tshape0*
T0*+
_output_shapes
:���������(
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
:���������(
�
lstm_1/BiasAdd_3BiasAddlstm_1/MatMul_3lstm_1/strided_slice_11*
data_formatNHWC*
T0*'
_output_shapes
:���������(
c
lstm_1/stack_3Const*
dtype0*!
valueB"����   (   *
_output_shapes
:
�
lstm_1/Reshape_7Reshapelstm_1/BiasAdd_3lstm_1/stack_3*
Tshape0*
T0*+
_output_shapes
:���������(
T
lstm_1/concat/axisConst*
dtype0*
value	B :*
_output_shapes
: 
�
lstm_1/concatConcatV2lstm_1/Reshape_1lstm_1/Reshape_3lstm_1/Reshape_5lstm_1/Reshape_7lstm_1/concat/axis*,
_output_shapes
:����������*

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
T0*,
_output_shapes
:����������
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
Index0*(
_output_shapes
:����������*

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
valueB"    (   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_15/stackConst*
dtype0*
valueB"    (   *
_output_shapes
:
p
lstm_1/strided_slice_15/stack_1Const*
dtype0*
valueB"    P   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_16/stackConst*
dtype0*
valueB"    P   *
_output_shapes
:
p
lstm_1/strided_slice_16/stack_1Const*
dtype0*
valueB"    x   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
n
lstm_1/strided_slice_17/stackConst*
dtype0*
valueB"    x   *
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
:���������(*

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
:���������(
�
lstm_1/MatMul_4MatMullstm_1/mul_1lstm_1/strided_slice_4*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
o
lstm_1/add_1Addlstm_1/strided_slice_14lstm_1/MatMul_4*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(

lstm_1/clip_by_valueMaximumlstm_1/clip_by_value/Minimumlstm_1/Const_2*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/MatMul_5MatMullstm_1/mul_3lstm_1/strided_slice_5*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
o
lstm_1/add_3Addlstm_1/strided_slice_15lstm_1/MatMul_5*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/clip_by_value_1Maximumlstm_1/clip_by_value_1/Minimumlstm_1/Const_4*
T0*'
_output_shapes
:���������(
j
lstm_1/mul_5Mullstm_1/clip_by_value_1lstm_1/Tile*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/MatMul_6MatMullstm_1/mul_6lstm_1/strided_slice_6*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
o
lstm_1/add_5Addlstm_1/strided_slice_16lstm_1/MatMul_6*
T0*'
_output_shapes
:���������(
S
lstm_1/TanhTanhlstm_1/add_5*
T0*'
_output_shapes
:���������(
h
lstm_1/mul_7Mullstm_1/clip_by_valuelstm_1/Tanh*
T0*'
_output_shapes
:���������(
a
lstm_1/add_6Addlstm_1/mul_5lstm_1/mul_7*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/MatMul_7MatMullstm_1/mul_8lstm_1/strided_slice_7*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
o
lstm_1/add_7Addlstm_1/strided_slice_17lstm_1/MatMul_7*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/clip_by_value_2Maximumlstm_1/clip_by_value_2/Minimumlstm_1/Const_6*
T0*'
_output_shapes
:���������(
U
lstm_1/Tanh_1Tanhlstm_1/add_6*
T0*'
_output_shapes
:���������(
m
lstm_1/mul_10Mullstm_1/clip_by_value_2lstm_1/Tanh_1*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/while/Enter_3Enterlstm_1/Tile*
parallel_iterations *
is_constant( *
T0**

frame_namelstm_1/while/lstm_1/while/*'
_output_shapes
:���������(
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
:���������(: 
�
lstm_1/while/Merge_3Mergelstm_1/while/Enter_3lstm_1/while/NextIteration_3*
N*
T0*)
_output_shapes
:���������(: 
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
&:���������(:���������(
�
lstm_1/while/Switch_3Switchlstm_1/while/Merge_3lstm_1/while/LoopCond*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:���������(:���������(
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
:���������(
n
lstm_1/while/Identity_3Identitylstm_1/while/Switch_3:1*
T0*'
_output_shapes
:���������(
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
loc:@lstm_1/TensorArray_1*(
_output_shapes
:����������
�
 lstm_1/while/strided_slice/stackConst^lstm_1/while/Identity*
dtype0*
valueB"        *
_output_shapes
:
�
"lstm_1/while/strided_slice/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    (   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_1/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    (   *
_output_shapes
:
�
$lstm_1/while/strided_slice_1/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    P   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_2/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    P   *
_output_shapes
:
�
$lstm_1/while/strided_slice_2/stack_1Const^lstm_1/while/Identity*
dtype0*
valueB"    x   *
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
:���������(*

begin_mask*
ellipsis_mask *
end_mask*
T0*
shrink_axis_mask 
�
"lstm_1/while/strided_slice_3/stackConst^lstm_1/while/Identity*
dtype0*
valueB"    x   *
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
:���������(*

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
:���������(
�
lstm_1/while/MatMul/EnterEnterlstm_1/strided_slice_4*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:((
�
lstm_1/while/MatMulMatMullstm_1/while/mullstm_1/while/MatMul/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
z
lstm_1/while/addAddlstm_1/while/strided_slicelstm_1/while/MatMul*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/while/clip_by_valueMaximum"lstm_1/while/clip_by_value/Minimumlstm_1/while/Const*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/while/MatMul_1/EnterEnterlstm_1/strided_slice_5*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:((
�
lstm_1/while/MatMul_1MatMullstm_1/while/mul_2lstm_1/while/MatMul_1/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
�
lstm_1/while/add_2Addlstm_1/while/strided_slice_1lstm_1/while/MatMul_1*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/while/clip_by_value_1Maximum$lstm_1/while/clip_by_value_1/Minimumlstm_1/while/Const_2*
T0*'
_output_shapes
:���������(
�
lstm_1/while/mul_4Mullstm_1/while/clip_by_value_1lstm_1/while/Identity_3*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/while/MatMul_2/EnterEnterlstm_1/strided_slice_6*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:((
�
lstm_1/while/MatMul_2MatMullstm_1/while/mul_5lstm_1/while/MatMul_2/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
�
lstm_1/while/add_4Addlstm_1/while/strided_slice_2lstm_1/while/MatMul_2*
T0*'
_output_shapes
:���������(
_
lstm_1/while/TanhTanhlstm_1/while/add_4*
T0*'
_output_shapes
:���������(
z
lstm_1/while/mul_6Mullstm_1/while/clip_by_valuelstm_1/while/Tanh*
T0*'
_output_shapes
:���������(
s
lstm_1/while/add_5Addlstm_1/while/mul_4lstm_1/while/mul_6*
T0*'
_output_shapes
:���������(
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
:���������(
�
lstm_1/while/MatMul_3/EnterEnterlstm_1/strided_slice_7*
parallel_iterations *
is_constant(*
T0**

frame_namelstm_1/while/lstm_1/while/*
_output_shapes

:((
�
lstm_1/while/MatMul_3MatMullstm_1/while/mul_7lstm_1/while/MatMul_3/Enter*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:���������(
�
lstm_1/while/add_6Addlstm_1/while/strided_slice_3lstm_1/while/MatMul_3*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
lstm_1/while/clip_by_value_2Maximum$lstm_1/while/clip_by_value_2/Minimumlstm_1/while/Const_4*
T0*'
_output_shapes
:���������(
a
lstm_1/while/Tanh_1Tanhlstm_1/while/add_5*
T0*'
_output_shapes
:���������(
~
lstm_1/while/mul_9Mullstm_1/while/clip_by_value_2lstm_1/while/Tanh_1*
T0*'
_output_shapes
:���������(
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
:���������(
s
lstm_1/while/NextIteration_3NextIterationlstm_1/while/add_5*
T0*'
_output_shapes
:���������(
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
:���������(
d
lstm_1/while/Exit_3Exitlstm_1/while/Switch_3*
T0*'
_output_shapes
:���������(
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
element_shape:���������(*
dtype0*%
_class
loc:@lstm_1/TensorArray*4
_output_shapes"
 :������������������(
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
:���������(
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
 :������������������(
e
lstm_1/Square_1Squarelstm_1/TensorArrayReadV3*
T0*'
_output_shapes
:���������(
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
:���������(
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
valueB"(      *
_output_shapes
:
_
dense_1/random_uniform/minConst*
dtype0*
valueB
 *�þ*
_output_shapes
: 
_
dense_1/random_uniform/maxConst*
dtype0*
valueB
 *��>*
_output_shapes
: 
�
$dense_1/random_uniform/RandomUniformRandomUniformdense_1/random_uniform/shape*
dtype0*
seed2���*
seed���)*
T0*
_output_shapes

:(
z
dense_1/random_uniform/subSubdense_1/random_uniform/maxdense_1/random_uniform/min*
T0*
_output_shapes
: 
�
dense_1/random_uniform/mulMul$dense_1/random_uniform/RandomUniformdense_1/random_uniform/sub*
T0*
_output_shapes

:(
~
dense_1/random_uniformAdddense_1/random_uniform/muldense_1/random_uniform/min*
T0*
_output_shapes

:(
�
dense_1/kernel
VariableV2*
dtype0*
shape
:(*
shared_name *
	container *
_output_shapes

:(
�
dense_1/kernel/AssignAssigndense_1/kerneldense_1/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:(
{
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes

:(
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
:���������(
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
valueB"   �   *
_output_shapes
:
�
gradients/lstm_1/Sum_grad/TileTile!gradients/lstm_1/Sum_grad/Reshape(gradients/lstm_1/Sum_grad/Tile/multiples*

Tmultiples0*
_class
loc:@lstm_1/Sum*
T0*
_output_shapes
:	�
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
:���������(
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
:���������(
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
:���������(
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
valueB"   �   *
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
:	�
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
:	�
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
:	�
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
:���������(
�
$gradients/lstm_1/Square_1_grad/mul_1Mul&gradients/lstm_1/mul_11_grad/Reshape_1"gradients/lstm_1/Square_1_grad/mul*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������(
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
:	�
�
"gradients/lstm_1/Square_grad/mul_1Mul#gradients/lstm_1/mul_grad/Reshape_1 gradients/lstm_1/Square_grad/mul* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes
:	�
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
:���������(
�
&gradients/dense_1/MatMul_grad/MatMul_1MatMullstm_1/TensorArrayReadV3$gradients/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*!
_class
loc:@dense_1/MatMul*
T0*
_output_shapes

:(
�
gradients/AddNAddN$gradients/lstm_1/Square_1_grad/mul_1$gradients/dense_1/MatMul_grad/MatMul*
N*"
_class
loc:@lstm_1/Square_1*
T0*'
_output_shapes
:���������(
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
:���������(
j
gradients/zeros_like_1	ZerosLikelstm_1/while/Exit_3*
T0*'
_output_shapes
:���������(
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
:���������(*4

frame_name&$gradients/lstm_1/while/lstm_1/while/*
parallel_iterations *
T0*&
_class
loc:@lstm_1/while/Exit_2*
is_constant( 
�
)gradients/lstm_1/while/Exit_3_grad/b_exitEntergradients/zeros_like_1*'
_output_shapes
:���������(*4

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
:���������(: 
�
-gradients/lstm_1/while/Switch_3_grad/b_switchMerge)gradients/lstm_1/while/Exit_3_grad/b_exit4gradients/lstm_1/while/Switch_3_grad_1/NextIteration*
N*'
_class
loc:@lstm_1/while/Merge_3*
T0*)
_output_shapes
:���������(: 
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
&:���������(:���������(
�
*gradients/lstm_1/while/Merge_3_grad/SwitchSwitch-gradients/lstm_1/while/Switch_3_grad/b_switchgradients/b_count_2*'
_class
loc:@lstm_1/while/Merge_3*
T0*:
_output_shapes(
&:���������(:���������(
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
:���������(
�
gradients/AddN_1AddN,gradients/lstm_1/while/Merge_2_grad/Switch:1Qgradients/lstm_1/while/TensorArrayWrite/TensorArrayWriteV3_grad/TensorArrayReadV3*
N*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:���������(
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
:���������(
�
%gradients/lstm_1/while/mul_9_grad/mulMulgradients/AddN_1.gradients/lstm_1/while/mul_9_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
�
'gradients/lstm_1/while/mul_9_grad/mul_1Mul0gradients/lstm_1/while/mul_9_grad/mul_1/StackPopgradients/AddN_1*%
_class
loc:@lstm_1/while/mul_9*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_2_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*'
_output_shapes
:���������(
�
4gradients/lstm_1/while/clip_by_value_2_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_2_grad/LogicalNot)gradients/lstm_1/while/mul_9_grad/Reshape1gradients/lstm_1/while/clip_by_value_2_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_2*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*'
_output_shapes
:���������(
�
<gradients/lstm_1/while/clip_by_value_2/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_2/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_2_grad/Reshape9gradients/lstm_1/while/clip_by_value_2/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_2/Minimum*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
%gradients/lstm_1/while/mul_4_grad/mulMul)gradients/lstm_1/while/add_5_grad/Reshape.gradients/lstm_1/while/mul_4_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
�
'gradients/lstm_1/while/mul_4_grad/mul_1Mul0gradients/lstm_1/while/mul_4_grad/mul_1/StackPop)gradients/lstm_1/while/add_5_grad/Reshape*%
_class
loc:@lstm_1/while/mul_4*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
�
%gradients/lstm_1/while/mul_6_grad/mulMul+gradients/lstm_1/while/add_5_grad/Reshape_1.gradients/lstm_1/while/mul_6_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
�
'gradients/lstm_1/while/mul_6_grad/mul_1Mul0gradients/lstm_1/while/mul_6_grad/mul_1/StackPop+gradients/lstm_1/while/add_5_grad/Reshape_1*%
_class
loc:@lstm_1/while/mul_6*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
�
%gradients/lstm_1/while/mul_8_grad/mulMul)gradients/lstm_1/while/add_7_grad/Reshape.gradients/lstm_1/while/mul_8_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_8*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot
LogicalNot8gradients/lstm_1/while/clip_by_value_1_grad/GreaterEqual*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*'
_output_shapes
:���������(
�
4gradients/lstm_1/while/clip_by_value_1_grad/Select_1Select6gradients/lstm_1/while/clip_by_value_1_grad/LogicalNot)gradients/lstm_1/while/mul_4_grad/Reshape1gradients/lstm_1/while/clip_by_value_1_grad/zeros*/
_class%
#!loc:@lstm_1/while/clip_by_value_1*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
4gradients/lstm_1/while/clip_by_value_grad/LogicalNot
LogicalNot6gradients/lstm_1/while/clip_by_value_grad/GreaterEqual*-
_class#
!loc:@lstm_1/while/clip_by_value*'
_output_shapes
:���������(
�
2gradients/lstm_1/while/clip_by_value_grad/Select_1Select4gradients/lstm_1/while/clip_by_value_grad/LogicalNot)gradients/lstm_1/while/mul_6_grad/Reshape/gradients/lstm_1/while/clip_by_value_grad/zeros*-
_class#
!loc:@lstm_1/while/clip_by_value*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot
LogicalNot=gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LessEqual*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*'
_output_shapes
:���������(
�
<gradients/lstm_1/while/clip_by_value_1/Minimum_grad/Select_1Select>gradients/lstm_1/while/clip_by_value_1/Minimum_grad/LogicalNot3gradients/lstm_1/while/clip_by_value_1_grad/Reshape9gradients/lstm_1/while/clip_by_value_1/Minimum_grad/zeros*7
_class-
+)loc:@lstm_1/while/clip_by_value_1/Minimum*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot
LogicalNot;gradients/lstm_1/while/clip_by_value/Minimum_grad/LessEqual*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*'
_output_shapes
:���������(
�
:gradients/lstm_1/while/clip_by_value/Minimum_grad/Select_1Select<gradients/lstm_1/while/clip_by_value/Minimum_grad/LogicalNot1gradients/lstm_1/while/clip_by_value_grad/Reshape7gradients/lstm_1/while/clip_by_value/Minimum_grad/zeros*5
_class+
)'loc:@lstm_1/while/clip_by_value/Minimum*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
Index0*(
_output_shapes
:����������*

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

:((*4

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
:���������(
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
:���������(
�
-gradients/lstm_1/while/MatMul_3_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_3_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_6_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_3*
T0*
_output_shapes

:((
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
:���������(
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
:���������(
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
Index0*(
_output_shapes
:����������*

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

:((*4

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
:���������(
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
:���������(
�
-gradients/lstm_1/while/MatMul_2_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_2_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_4_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_2*
T0*
_output_shapes

:((
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
:���������(
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
:���������(
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
:���������(
�
'gradients/lstm_1/while/mul_7_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_3_grad/MatMul*%
_class
loc:@lstm_1/while/mul_7*
T0*'
_output_shapes
:���������(
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
valueB((*    *
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc*
_output_shapes

:((*4

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
:((: 
�
1gradients/lstm_1/while/MatMul_3/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*(
_output_shapes
:((:((
�
.gradients/lstm_1/while/MatMul_3/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_3/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_3_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:((
�
8gradients/lstm_1/while/MatMul_3/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_3/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_3/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_3/Enter*
T0*
_output_shapes

:((
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
:���������(
�
%gradients/lstm_1/while/mul_3_grad/mulMul)gradients/lstm_1/while/add_3_grad/Reshape.gradients/lstm_1/while/mul_3_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_3*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
�
%gradients/lstm_1/while/mul_1_grad/mulMul)gradients/lstm_1/while/add_1_grad/Reshape.gradients/lstm_1/while/mul_1_grad/mul/StackPop*%
_class
loc:@lstm_1/while/mul_1*
T0*'
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
'gradients/lstm_1/while/mul_5_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_2_grad/MatMul*%
_class
loc:@lstm_1/while/mul_5*
T0*'
_output_shapes
:���������(
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
valueB((*    *
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc*
_output_shapes

:((*4

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
:((: 
�
1gradients/lstm_1/while/MatMul_2/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*(
_output_shapes
:((:((
�
.gradients/lstm_1/while/MatMul_2/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_2/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_2_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:((
�
8gradients/lstm_1/while/MatMul_2/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_2/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_2/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_2/Enter*
T0*
_output_shapes

:((
�
+gradients/lstm_1/strided_slice_7_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_7*
valueB"(   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_7_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_7_grad/Shapelstm_1/strided_slice_7/stacklstm_1/strided_slice_7/stack_1lstm_1/strided_slice_7/stack_22gradients/lstm_1/while/MatMul_3/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes
:	(�*

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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
+gradients/lstm_1/strided_slice_6_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_6*
valueB"(   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_6_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_6_grad/Shapelstm_1/strided_slice_6/stacklstm_1/strided_slice_6/stack_1lstm_1/strided_slice_6/stack_22gradients/lstm_1/while/MatMul_2/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes
:	(�*

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
Index0*(
_output_shapes
:����������*

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

:((*4

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
:���������(
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
:���������(
�
-gradients/lstm_1/while/MatMul_1_grad/MatMul_1MatMul6gradients/lstm_1/while/MatMul_1_grad/MatMul_1/StackPop+gradients/lstm_1/while/add_2_grad/Reshape_1*
transpose_b( *
transpose_a(*(
_class
loc:@lstm_1/while/MatMul_1*
T0*
_output_shapes

:((
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
Index0*(
_output_shapes
:����������*

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

:((*4

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
:���������(
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
:���������(
�
+gradients/lstm_1/while/MatMul_grad/MatMul_1MatMul4gradients/lstm_1/while/MatMul_grad/MatMul_1/StackPop)gradients/lstm_1/while/add_grad/Reshape_1*
transpose_b( *
transpose_a(*&
_class
loc:@lstm_1/while/MatMul*
T0*
_output_shapes

:((
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
:���������(
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
:���������(
�
'gradients/lstm_1/while/mul_2_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop+gradients/lstm_1/while/MatMul_1_grad/MatMul*%
_class
loc:@lstm_1/while/mul_2*
T0*'
_output_shapes
:���������(
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
valueB((*    *
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_1Enter0gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc*
_output_shapes

:((*4

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
:((: 
�
1gradients/lstm_1/while/MatMul_1/Enter_grad/SwitchSwitch2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_2gradients/b_count_2*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*(
_output_shapes
:((:((
�
.gradients/lstm_1/while/MatMul_1/Enter_grad/AddAdd3gradients/lstm_1/while/MatMul_1/Enter_grad/Switch:1-gradients/lstm_1/while/MatMul_1_grad/MatMul_1*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:((
�
8gradients/lstm_1/while/MatMul_1/Enter_grad/NextIterationNextIteration.gradients/lstm_1/while/MatMul_1/Enter_grad/Add*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:((
�
2gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3Exit1gradients/lstm_1/while/MatMul_1/Enter_grad/Switch*.
_class$
" loc:@lstm_1/while/MatMul_1/Enter*
T0*
_output_shapes

:((
�
gradients/AddN_3AddN<gradients/lstm_1/while/strided_slice_3_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_2_grad/StridedSliceGrad<gradients/lstm_1/while/strided_slice_1_grad/StridedSliceGrad:gradients/lstm_1/while/strided_slice_grad/StridedSliceGrad*
N*/
_class%
#!loc:@lstm_1/while/strided_slice_3*
T0*(
_output_shapes
:����������
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
:���������(
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
:���������(
�
%gradients/lstm_1/while/mul_grad/mul_1Mul0gradients/lstm_1/while/mul_7_grad/mul_1/StackPop)gradients/lstm_1/while/MatMul_grad/MatMul*#
_class
loc:@lstm_1/while/mul*
T0*'
_output_shapes
:���������(
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
valueB((*    *
_output_shapes

:((
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_1Enter.gradients/lstm_1/while/MatMul/Enter_grad/b_acc*
_output_shapes

:((*4

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
:((: 
�
/gradients/lstm_1/while/MatMul/Enter_grad/SwitchSwitch0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_2gradients/b_count_2*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*(
_output_shapes
:((:((
�
,gradients/lstm_1/while/MatMul/Enter_grad/AddAdd1gradients/lstm_1/while/MatMul/Enter_grad/Switch:1+gradients/lstm_1/while/MatMul_grad/MatMul_1*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:((
�
6gradients/lstm_1/while/MatMul/Enter_grad/NextIterationNextIteration,gradients/lstm_1/while/MatMul/Enter_grad/Add*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:((
�
0gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3Exit/gradients/lstm_1/while/MatMul/Enter_grad/Switch*,
_class"
 loc:@lstm_1/while/MatMul/Enter*
T0*
_output_shapes

:((
�
+gradients/lstm_1/strided_slice_5_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_5*
valueB"(   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_5_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_5_grad/Shapelstm_1/strided_slice_5/stacklstm_1/strided_slice_5/stack_1lstm_1/strided_slice_5/stack_22gradients/lstm_1/while/MatMul_1/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes
:	(�*

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
:���������(
�
+gradients/lstm_1/strided_slice_4_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_4*
valueB"(   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_4_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_4_grad/Shapelstm_1/strided_slice_4/stacklstm_1/strided_slice_4/stack_1lstm_1/strided_slice_4/stack_20gradients/lstm_1/while/MatMul/Enter_grad/b_acc_3*
new_axis_mask *
Index0*
_output_shapes
:	(�*

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
loc:@lstm_1/TensorArray_1*,
_output_shapes
:����������
�
4gradients/lstm_1/while/Switch_2_grad_1/NextIterationNextIterationgradients/AddN_4*'
_class
loc:@lstm_1/while/Merge_2*
T0*'
_output_shapes
:���������(
�
gradients/AddN_5AddN6gradients/lstm_1/strided_slice_7_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_6_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_5_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_4_grad/StridedSliceGrad*
N*)
_class
loc:@lstm_1/strided_slice_7*
T0*
_output_shapes
:	(�
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
T0*,
_output_shapes
:����������
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
:���������(
�
$gradients/lstm_1/concat_grad/Slice_1Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:1%gradients/lstm_1/concat_grad/ShapeN:1*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������(
�
$gradients/lstm_1/concat_grad/Slice_2Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:2%gradients/lstm_1/concat_grad/ShapeN:2*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������(
�
$gradients/lstm_1/concat_grad/Slice_3Slice)gradients/lstm_1/transpose_grad/transpose+gradients/lstm_1/concat_grad/ConcatOffset:3%gradients/lstm_1/concat_grad/ShapeN:3*
Index0* 
_class
loc:@lstm_1/concat*
T0*+
_output_shapes
:���������(
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
:���������(
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
:���������(
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
:���������(
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
:���������(
�
)gradients/lstm_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_1_grad/Reshape*!
_class
loc:@lstm_1/BiasAdd*
data_formatNHWC*
T0*
_output_shapes
:(
�
+gradients/lstm_1/BiasAdd_1_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_3_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_1*
data_formatNHWC*
T0*
_output_shapes
:(
�
+gradients/lstm_1/BiasAdd_2_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_5_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_2*
data_formatNHWC*
T0*
_output_shapes
:(
�
+gradients/lstm_1/BiasAdd_3_grad/BiasAddGradBiasAddGrad'gradients/lstm_1/Reshape_7_grad/Reshape*#
_class
loc:@lstm_1/BiasAdd_3*
data_formatNHWC*
T0*
_output_shapes
:(
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

:(
�
+gradients/lstm_1/strided_slice_8_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_8*
valueB:�*
_output_shapes
:
�
6gradients/lstm_1/strided_slice_8_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_8_grad/Shapelstm_1/strided_slice_8/stacklstm_1/strided_slice_8/stack_1lstm_1/strided_slice_8/stack_2)gradients/lstm_1/BiasAdd_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes	
:�*

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

:(
�
+gradients/lstm_1/strided_slice_9_grad/ShapeConst*
dtype0*)
_class
loc:@lstm_1/strided_slice_9*
valueB:�*
_output_shapes
:
�
6gradients/lstm_1/strided_slice_9_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_9_grad/Shapelstm_1/strided_slice_9/stacklstm_1/strided_slice_9/stack_1lstm_1/strided_slice_9/stack_2+gradients/lstm_1/BiasAdd_1_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes	
:�*

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

:(
�
,gradients/lstm_1/strided_slice_10_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_10*
valueB:�*
_output_shapes
:
�
7gradients/lstm_1/strided_slice_10_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_10_grad/Shapelstm_1/strided_slice_10/stacklstm_1/strided_slice_10/stack_1lstm_1/strided_slice_10/stack_2+gradients/lstm_1/BiasAdd_2_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes	
:�*

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

:(
�
,gradients/lstm_1/strided_slice_11_grad/ShapeConst*
dtype0**
_class 
loc:@lstm_1/strided_slice_11*
valueB:�*
_output_shapes
:
�
7gradients/lstm_1/strided_slice_11_grad/StridedSliceGradStridedSliceGrad,gradients/lstm_1/strided_slice_11_grad/Shapelstm_1/strided_slice_11/stacklstm_1/strided_slice_11/stack_1lstm_1/strided_slice_11/stack_2+gradients/lstm_1/BiasAdd_3_grad/BiasAddGrad*
new_axis_mask *
Index0*
_output_shapes	
:�*

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
valueB"   �   *
_output_shapes
:
�
4gradients/lstm_1/strided_slice_grad/StridedSliceGradStridedSliceGrad)gradients/lstm_1/strided_slice_grad/Shapelstm_1/strided_slice/stacklstm_1/strided_slice/stack_1lstm_1/strided_slice/stack_2%gradients/lstm_1/MatMul_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes
:	�*

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
valueB"   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_1_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_1_grad/Shapelstm_1/strided_slice_1/stacklstm_1/strided_slice_1/stack_1lstm_1/strided_slice_1/stack_2'gradients/lstm_1/MatMul_1_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes
:	�*

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
valueB"   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_2_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_2_grad/Shapelstm_1/strided_slice_2/stacklstm_1/strided_slice_2/stack_1lstm_1/strided_slice_2/stack_2'gradients/lstm_1/MatMul_2_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes
:	�*

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
valueB"   �   *
_output_shapes
:
�
6gradients/lstm_1/strided_slice_3_grad/StridedSliceGradStridedSliceGrad+gradients/lstm_1/strided_slice_3_grad/Shapelstm_1/strided_slice_3/stacklstm_1/strided_slice_3/stack_1lstm_1/strided_slice_3/stack_2'gradients/lstm_1/MatMul_3_grad/MatMul_1*
new_axis_mask *
Index0*
_output_shapes
:	�*

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
T0*
_output_shapes	
:�
�
gradients/AddN_7AddN"gradients/lstm_1/Square_grad/mul_14gradients/lstm_1/strided_slice_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_1_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_2_grad/StridedSliceGrad6gradients/lstm_1/strided_slice_3_grad/StridedSliceGrad*
N* 
_class
loc:@lstm_1/Square*
T0*
_output_shapes
:	�
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
valueB(*    *
_output_shapes

:(
~

Variable_1
VariableV2*
dtype0*
shape
:(*
shared_name *
	container *
_output_shapes

:(
�
Variable_1/AssignAssign
Variable_1Const_3*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:(
o
Variable_1/readIdentity
Variable_1*
_class
loc:@Variable_1*
T0*
_output_shapes

:(
V
Const_4Const*
dtype0*
valueB�*    *
_output_shapes	
:�
x

Variable_2
VariableV2*
dtype0*
shape:�*
shared_name *
	container *
_output_shapes	
:�
�
Variable_2/AssignAssign
Variable_2Const_4*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes	
:�
l
Variable_2/readIdentity
Variable_2*
_class
loc:@Variable_2*
T0*
_output_shapes	
:�
^
Const_5Const*
dtype0*
valueB	�*    *
_output_shapes
:	�
�

Variable_3
VariableV2*
dtype0*
shape:	�*
shared_name *
	container *
_output_shapes
:	�
�
Variable_3/AssignAssign
Variable_3Const_5*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:	�
p
Variable_3/readIdentity
Variable_3*
_class
loc:@Variable_3*
T0*
_output_shapes
:	�
^
Const_6Const*
dtype0*
valueB	(�*    *
_output_shapes
:	(�
�

Variable_4
VariableV2*
dtype0*
shape:	(�*
shared_name *
	container *
_output_shapes
:	(�
�
Variable_4/AssignAssign
Variable_4Const_6*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes
:	(�
p
Variable_4/readIdentity
Variable_4*
_class
loc:@Variable_4*
T0*
_output_shapes
:	(�
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

:(
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

:(
F
mul_6Mulsub_3Square_2*
T0*
_output_shapes

:(
C
add_4Addmul_5mul_6*
T0*
_output_shapes

:(
�
Assign_2Assign
Variable_1add_4*
validate_shape(*
_class
loc:@Variable_1*
use_locking(*
T0*
_output_shapes

:(
f
mul_7Mullr/read&gradients/dense_1/MatMul_grad/MatMul_1*
T0*
_output_shapes

:(
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

:(
e
clip_by_value_1Maximumclip_by_value_1/MinimumConst_9*
T0*
_output_shapes

:(
H
Sqrt_1Sqrtclip_by_value_1*
T0*
_output_shapes

:(
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

:(
G
div_2RealDivmul_7add_5*
T0*
_output_shapes

:(
Q
sub_4Subdense_1/kernel/readdiv_2*
T0*
_output_shapes

:(
�
Assign_3Assigndense_1/kernelsub_4*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0*
_output_shapes

:(
M
mul_8Mulrho/readVariable_2/read*
T0*
_output_shapes	
:�
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
J
Square_3Squaregradients/AddN_6*
T0*
_output_shapes	
:�
C
mul_9Mulsub_5Square_3*
T0*
_output_shapes	
:�
@
add_6Addmul_8mul_9*
T0*
_output_shapes	
:�
�
Assign_4Assign
Variable_2add_6*
validate_shape(*
_class
loc:@Variable_2*
use_locking(*
T0*
_output_shapes	
:�
N
mul_10Mullr/readgradients/AddN_6*
T0*
_output_shapes	
:�
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
Y
clip_by_value_2/MinimumMinimumadd_6Const_12*
T0*
_output_shapes	
:�
c
clip_by_value_2Maximumclip_by_value_2/MinimumConst_11*
T0*
_output_shapes	
:�
E
Sqrt_2Sqrtclip_by_value_2*
T0*
_output_shapes	
:�
L
add_7/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
C
add_7AddSqrt_2add_7/y*
T0*
_output_shapes	
:�
E
div_3RealDivmul_10add_7*
T0*
_output_shapes	
:�
K
sub_6Sublstm_1/bias/readdiv_3*
T0*
_output_shapes	
:�
�
Assign_5Assignlstm_1/biassub_6*
validate_shape(*
_class
loc:@lstm_1/bias*
use_locking(*
T0*
_output_shapes	
:�
R
mul_11Mulrho/readVariable_3/read*
T0*
_output_shapes
:	�
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
N
Square_4Squaregradients/AddN_7*
T0*
_output_shapes
:	�
H
mul_12Mulsub_7Square_4*
T0*
_output_shapes
:	�
F
add_8Addmul_11mul_12*
T0*
_output_shapes
:	�
�
Assign_6Assign
Variable_3add_8*
validate_shape(*
_class
loc:@Variable_3*
use_locking(*
T0*
_output_shapes
:	�
R
mul_13Mullr/readgradients/AddN_7*
T0*
_output_shapes
:	�
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
]
clip_by_value_3/MinimumMinimumadd_8Const_14*
T0*
_output_shapes
:	�
g
clip_by_value_3Maximumclip_by_value_3/MinimumConst_13*
T0*
_output_shapes
:	�
I
Sqrt_3Sqrtclip_by_value_3*
T0*
_output_shapes
:	�
L
add_9/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
G
add_9AddSqrt_3add_9/y*
T0*
_output_shapes
:	�
I
div_4RealDivmul_13add_9*
T0*
_output_shapes
:	�
Q
sub_8Sublstm_1/kernel/readdiv_4*
T0*
_output_shapes
:	�
�
Assign_7Assignlstm_1/kernelsub_8*
validate_shape(* 
_class
loc:@lstm_1/kernel*
use_locking(*
T0*
_output_shapes
:	�
R
mul_14Mulrho/readVariable_4/read*
T0*
_output_shapes
:	(�
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
N
Square_5Squaregradients/AddN_5*
T0*
_output_shapes
:	(�
H
mul_15Mulsub_9Square_5*
T0*
_output_shapes
:	(�
G
add_10Addmul_14mul_15*
T0*
_output_shapes
:	(�
�
Assign_8Assign
Variable_4add_10*
validate_shape(*
_class
loc:@Variable_4*
use_locking(*
T0*
_output_shapes
:	(�
R
mul_16Mullr/readgradients/AddN_5*
T0*
_output_shapes
:	(�
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
^
clip_by_value_4/MinimumMinimumadd_10Const_16*
T0*
_output_shapes
:	(�
g
clip_by_value_4Maximumclip_by_value_4/MinimumConst_15*
T0*
_output_shapes
:	(�
I
Sqrt_4Sqrtclip_by_value_4*
T0*
_output_shapes
:	(�
M
add_11/yConst*
dtype0*
valueB
 *w�+2*
_output_shapes
: 
I
add_11AddSqrt_4add_11/y*
T0*
_output_shapes
:	(�
J
div_5RealDivmul_16add_11*
T0*
_output_shapes
:	(�
\
sub_10Sublstm_1/recurrent_kernel/readdiv_5*
T0*
_output_shapes
:	(�
�
Assign_9Assignlstm_1/recurrent_kernelsub_10*
validate_shape(**
_class 
loc:@lstm_1/recurrent_kernel*
use_locking(*
T0*
_output_shapes
:	(�
�
group_deps_1NoOp^add_1^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
�
initNoOp^dense_1/kernel/Assign^dense_1/bias/Assign
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
Variable_4:0Variable_4/AssignVariable_4/read:0 �0       �K"	S�7DI�A*

loss�W=�y9
       Oa�(	�U�7DI�A*
	
lr
�#<���       ���	X�7DI�A*

val_loss�<D
�d       ��-	�+98DI�A*

loss��<�Ҥ       �K"	.98DI�A*
	
lr
�#<M�2       ��2	�/98DI�A*

val_loss2v�<���       ��-	|�8DI�A*

loss��<UH�       �K"	��8DI�A*
	
lr
�#<>�V       ��2	b�8DI�A*

val_loss�=H       ��-	��9DI�A*

loss�֎<�=�y       �K"	Â9DI�A*
	
lr
�#<ǧ��       ��2	{�9DI�A*

val_loss���<䑖{       ��-	�y9DI�A*

loss�|�<R��R       �K"	5�y9DI�A*
	
lr
�#<��@]       ��2	=�y9DI�A*

val_lossX�<�=�E       ��-	Z��9DI�A*

loss^)y<�d�       �K"	o��9DI�A*
	
lr
�#<[�       ��2	#��9DI�A*

val_lossF˿<b���       ��-	�
O:DI�A*

loss�w<C@�       �K"	�O:DI�A*
	
lr
�#<���W       ��2	�O:DI�A*

val_loss$y`<~#h�       ��-	��:DI�A*

loss��X<�+3       �K"	��:DI�A*
	
lr
�#<�r�b       ��2	��:DI�A*

val_lossM=��
�       ��-	�;DI�A*

loss�[i<�P(�       �K"	/�;DI�A*
	
lr
�#<R-Pq       ��2	�;DI�A*

val_loss{э< se�       ��-	h�;DI�A	*

loss�P<�
�       �K"	��;DI�A	*
	
lr
�#<��s       ��2	��;DI�A	*

val_loss�Ԓ<����       ��-	�~�;DI�A
*

loss%�\<O�#       �K"	���;DI�A
*
	
lr
�#<K=       ��2	k��;DI�A
*

val_loss}'=���       ��-	�C`<DI�A*

lossȼJ<�{f	       �K"	�E`<DI�A*
	
lr
�#<6��+       ��2	|G`<DI�A*

val_loss�4�<�U�       ��-	�H�<DI�A*

lossAuK<�{��       �K"	�J�<DI�A*
	
lr
�#<��MU       ��2	�L�<DI�A*

val_loss��V<��FW       ��-	�6=DI�A*

lossV�;<��_�       �K"	�6=DI�A*
	
lr
�#<�>>�       ��2	;�6=DI�A*

val_lossM�D<d��2       ��-	��=DI�A*

loss�nE<IIm�       �K"	���=DI�A*
	
lr
�#<�4�E       ��2	b��=DI�A*

val_losso��=���1       ��-	A�>DI�A*

loss��b<�cح       �K"	Z�>DI�A*
	
lr
�#<��T�       ��2	@�>DI�A*

val_lossG��<����       ��-	̶u>DI�A*

loss�'<)�l       �K"	ظu>DI�A*
	
lr
�#<6��       ��2	úu>DI�A*

val_loss�+=)%�Z       ��-	���>DI�A*

loss0<9<�ۓh       �K"	��>DI�A*
	
lr
�#<Pj@]       ��2	���>DI�A*

val_loss!L�<+��       ��-	B?K?DI�A*

loss�4<W��       �K"	WAK?DI�A*
	
lr
�#<W�D       ��2	=CK?DI�A*

val_loss��<�c�       ��-	���?DI�A*

loss�d6<A$\�       �K"	ղ�?DI�A*
	
lr
�#<,^>       ��2	���?DI�A*

val_loss�̑<���x       ��-	�M @DI�A*

loss�54<�V��       �K"	�O @DI�A*
	
lr
�#<�=��       ��2	�Q @DI�A*

val_loss���<(�M       ��-	�Ŋ@DI�A*

loss��3<>��       �K"	�Ǌ@DI�A*
	
lr
�#<��4       ��2	~Ɋ@DI�A*

val_loss&4=���       ��-	,I�@DI�A*

loss��><�WP^       �K"	@K�@DI�A*
	
lr
�#<7ӹ#       ��2	'M�@DI�A*

val_loss�w�<Ӕ8m       ��-	�^ADI�A*

loss��0<���<       �K"	;�^ADI�A*
	
lr
�#<�7��       ��2	%�^ADI�A*

val_loss��u<BN��       ��-	�N�ADI�A*

loss�)<���       �K"	�R�ADI�A*
	
lr
�#<�h;       ��2	~V�ADI�A*

val_loss�c="�:       ��-	�6BDI�A*

loss�(<���       �K"	�6BDI�A*
	
lrn;���       ��2	��6BDI�A*

val_lossH�<��       ��-	=H�BDI�A*

loss�<���5       �K"	VJ�BDI�A*
	
lrn;�`�       ��2	L�BDI�A*

val_loss:!w<�b8,       ��-	�	CDI�A*

lossG�<��S       �K"	�CDI�A*
	
lrn;��u       ��2	�CDI�A*

val_loss-�<�P�y       ��-	��vCDI�A*

lossx�<��       �K"	ҨvCDI�A*
	
lrn;�؍�       ��2	��vCDI�A*

val_loss:�k<p��       ��-	Yk�CDI�A*

loss�6<���U       �K"	m�CDI�A*
	
lrn;��J�       ��2	ro�CDI�A*

val_loss��<���