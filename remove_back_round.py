import SimpleITK as sitk
import torch
from common import *



def read_med_image(file_path, dtype):
    img_stk = sitk.ReadImage(file_path)
    img_np = sitk.GetArrayFromImage(img_stk)
    img_np = img_np.astype(dtype)
    # plt.imshow(img_np)
    # plt.show()
    return img_np, img_stk
test_path = './iSeg-2019-Testing'

for subject_id in range(24, 40):
    subject_name = 'subject-%d-' % subject_id
    f_T1 = os.path.join(test_path, subject_name + 'T1.hdr')
    f_T1_org = os.path.join(test_path, subject_name + 'label.hdr')

    inputs_T1, img_T1_itk = read_med_image(f_T1, dtype=np.float32)
    inputs_T1_org, img_T1_itk_org = read_med_image(f_T1_org, dtype=np.float32)
    print(inputs_T1_org.shape, '----')

    mask = inputs_T1 > 0
    mask = mask.astype(np.bool)

    new_seg = inputs_T1_org * mask

    f_pred = os.path.join(test_path, subject_name + 'label-trung.hdr')
    whole_pred_itk = sitk.GetImageFromArray(new_seg.astype(np.uint8))
    whole_pred_itk.SetSpacing(img_T1_itk.GetSpacing())
    whole_pred_itk.SetDirection(img_T1_itk.GetDirection())
    sitk.WriteImage(whole_pred_itk, f_pred)