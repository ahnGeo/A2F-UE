# @Geo Ahn
# Add a lot of curves' key values (2d array) by unreal-python API

import unreal
import pickle as pc

anim_lib = unreal.AnimationLibrary()
face_anim = unreal.EditorAssetLibrary.load_asset("Animation sequence using face mesh path")

with open("output_blendshape_weight_list.pkl","rb") as f:
    weight_list = pc.load(f)    # curve key values, each frame X blendshape weight
with open("curves_list_audio_6_mh_arkit.pkl","rb") as f:
    curves_value_name = pc.load(f)  # curve mapping information, i_th row is composed of tuples (mapping weight, curve name) mapped with i_th blendshape

# initialize curve data
anim_lib.remove_all_curve_data(face_anim)   
curves_name_list = ['CTRL_expressions_jawChinRaiseUR', 'head_wm1_normal_head_wm13_lips_UR',     #set(all used curves)
'head_cm2_color_head_wm2_browsLateral_L', 'CTRL_expressions_mouthDimpleL', 'head_wm1_normal_head_wm1_browsRaiseOuter_L', 'head_wm3_normal_head_wm13_lips_UL', 'CTRL_expressions_jawLeft', 'CTRL_expressions_mouthCheekBlowR', 'head_wm2_normal_head_wm2_browsLateral_R', 'CTRL_expressions_mouthUpperLipRollInL', 'head_cm1_color_head_wm1_browsRaiseOuter_L', 'CTRL_expressions_mouthDimpleR', 'head_wm1_normal_head_wm1_chinRaise_L', 'CTRL_expressions_mouthUpperLipRaiseR', 'CTRL_expressions_eyeLookRightR', 'CTRL_expressions_mouthLipsPurseDL', 'CTRL_expressions_browRaiseOuterL', 'CTRL_expressions_jawOpen', 'head_cm1_color_head_wm1_browsRaiseOuter_R', 'CTRL_expressions_eyeLookDownL', 'CTRL_expressions_jawRight', 'head_cm2_color_head_wm2_noseWrinkler_L', 'CTRL_expressions_mouthLeft', 'CTRL_expressions_eyeSquintInnerR', 'CTRL_expressions_mouthLowerLipRollInL', 'head_wm2_normal_head_wm2_noseWrinkler_L', 'head_wm2_normal_head_wm2_noseWrinkler_R', 'head_wm3_color_head_wm13_lips_DL', 'head_wm2_normal_head_wm2_browsDown_L', 'head_cm2_color_head_wm2_browsLateral_R', 'head_wm2_normal_head_wm2_browsLateral_L', 'head_cm1_color_head_wm1_blink_L', 'head_wm3_normal_head_wm13_lips_DL', 'head_wm1_normal_head_wm1_browsRaiseInner_L', 'CTRL_expressions_browRaiseOuterR', 'CTRL_expressions_eyeWidenL', 'head_cm2_color_head_wm2_browsDown_R', 'CTRL_expressions_mouthPressUR', 'CTRL_expressions_eyeLookLeftR', 'CTRL_expressions_mouthLipsBlowL', 'head_wm2_normal_head_wm2_mouthStretch_L', 'CTRL_expressions_mouthLipsPurseUL', 'head_cm1_color_head_wm13_lips_UR', 'head_wm1_normal_head_wm1_blink_R', 'CTRL_expressions_mouthPressUL', 'CTRL_expressions_mouthLipsPurseUR', 'head_cm2_color_head_wm2_noseWrinkler_R', 'head_cm2_color_head_wm2_mouthStretch_R', 'head_wm1_normal_head_wm1_jawOpen', 'CTRL_expressions_mouthRight', 'head_wm3_color_head_wm13_lips_UL', 'CTRL_expressions_eyeSquintInnerL', 'CTRL_expressions_eyeLookRightL', 'CTRL_expressions_mouthCornerPullR', 'CTRL_expressions_mouthCornerDepressR', 'CTRL_expressions_mouthLipsPurseDR', 'CTRL_expressions_eyeLookDownR', 'head_cm1_color_head_wm1_chinRaise_L', 'head_cm1_color_head_wm1_jawOpen', 'CTRL_expressions_mouthLowerLipDepressL', 'head_cm3_color_head_wm3_smile_L', 'CTRL_expressions_eyeCheekRaiseL', 'CTRL_expressions_jawFwd', 'CTRL_expressions_mouthUpperLipRollInR', 'CTRL_expressions_noseWrinkleL', 'CTRL_expressions_mouthStretchR', 'CTRL_expressions_browDownR', 'head_wm3_normal_head_wm3_smile_R', 'head_cm1_color_head_wm13_lips_UL', 'CTRL_expressions_eyeWidenR', 'CTRL_expressions_jawChinRaiseUL', 'CTRL_expressions_mouthCornerDepressL', 'head_cm1_color_head_wm1_browsRaiseInner_L', 'head_wm1_normal_head_wm1_blink_L', 'head_wm1_normal_head_wm1_browsRaiseInner_R', 'CTRL_expressions_mouthLipsBlowR', 'head_wm1_normal_head_wm13_lips_UL', 'CTRL_expressions_browRaiseInL', 'CTRL_expressions_mouthPressDR', 'head_wm1_normal_head_wm1_browsRaiseOuter_R', 'head_wm2_normal_head_wm2_mouthStretch_R', 'CTRL_expressions_mouthPressDL', 'CTRL_expressions_browLateralR', 'CTRL_expressions_eyeBlinkR', 'CTRL_expressions_mouthLowerLipDepressR', 'head_wm2_normal_head_wm2_browsDown_R', 'CTRL_expressions_noseWrinkleR', 'CTRL_expressions_jawChinRaiseDL', 'head_wm3_normal_head_wm3_smile_L', 'CTRL_expressions_mouthCheekBlowL', 'head_cm1_color_head_wm1_browsRaiseInner_R', 'CTRL_expressions_mouthCornerPullL', 'CTRL_expressions_eyeCheekRaiseR', 'CTRL_expressions_browRaiseInR', 'head_cm1_color_head_wm1_chinRaise_R', 'CTRL_expressions_eyeLookLeftL', 'head_cm1_color_head_wm1_blink_R', 'CTRL_expressions_jawChinRaiseDR', 'CTRL_expressions_mouthLowerLipRollInR', 'head_cm3_color_head_wm3_smile_R', 'head_cm2_color_head_wm2_browsDown_L', 'head_cm2_color_head_wm2_mouthStretch_L', 'CTRL_expressions_browDownL', 'CTRL_expressions_eyeLookUpL', 'CTRL_expressions_eyeLookUpR', 'CTRL_expressions_eyeBlinkL', 'CTRL_expressions_mouthUpperLipRaiseL', 'head_wm1_normal_head_wm1_chinRaise_R', 'CTRL_expressions_mouthStretchL', 'CTRL_expressions_browLateralL']
for elem in curves_name_list :
    anim_lib.add_curve(face_anim, elem)

len_frame = len(weight_list)
frame_rate = 60
times_array = [num_frame/frame_rate for num_frame in range(0, len_frame)]
for i in range(0, len(weight_list)) : 
    values_array = [float(row[i]) for row in weight_list]
    for each_curve in curves_value_name[i] :    #each_curve = (mapping weight, curve name)
        #print(each_curve, i)
        multiple_array = [each_curve[0] * value for value in values_array]
        anim_lib.add_float_curve_keys(face_anim, each_curve[1], times_array, multiple_array)
            