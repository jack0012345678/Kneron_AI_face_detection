from PIL import Image,ImageTk
import tkinter as tk
import tkinter.font as tkFont
import argparse
import pkgutil
import importlib
import os
import re
from common import kdp_wrapper
from common import constants, kdp_wrapper
from common.pre_post_process.yolo.yolo_postprocess import yolo_postprocess_
import cv2
import kdp_host_api as api
import sys
from datetime import datetime
''' kdp_host_api config '''
KDP_UART_DEV = 0
KDP_USB_DEV = 1
VENDOR_ID = 0x3231
KL520_PRODUCT_ID = 0x0100



''' import ignore list '''
IGNORE_MODULE_LIST = [
    'utils',
    'fdssd',
    'yolo',
    'keras_only',
    'fdr'
]


def get_module_names(examples_dir, example_regex=r'^examples_kl\d{3}'):
    module_names = []
    folder_names = [name for name in os.listdir(examples_dir) if os.path.isdir(name)]
    for folder_name in folder_names:
        if re.match(pattern=example_regex, string=folder_name):
            module_names.append(folder_name)
    return module_names


def get_all_module_path(package_path_list):
    example_dict = {}

    for package_path in package_path_list:
        for finder, name, _ in pkgutil.iter_modules(path=[package_path]):
            package_name = os.path.basename(finder.path)
            search_obj = re.search(pattern=r'(kl\d\d\d)$', string=package_name)

            if search_obj and (name not in IGNORE_MODULE_LIST):
                kneron_device_name = search_obj.group(1)
                example_dict['-'.join([kneron_device_name.upper(), name])] = '{}.{}'.format(package_name, name)
    return example_dict


def import_example_function(module_path, function_name):
    try:
        module = importlib.import_module(module_path)
        if hasattr(module, function_name):
            return getattr(module, function_name)
    except Exception as e:
        print('Can not import {}, function {}'.format(module_path, function_name))
def end_det(dev_idx):
    """Ends DME mode for the specified device."""
    api.kdp_end_dme(dev_idx)

''' get example modules '''
example_module_folder_path = os.path.dirname(os.path.abspath(__file__))
module_names = get_module_names(examples_dir=example_module_folder_path,
                                example_regex=r'^examples_kl\d\d\d$')


example_dict = get_all_module_path(
    package_path_list=[os.path.join(example_module_folder_path, module_name) for module_name in module_names])


''' input parameters '''
argparser = argparse.ArgumentParser(
    description="Run Python examples by calling the Python APIs",
    formatter_class=argparse.RawTextHelpFormatter)

argparser.add_argument(
    '-t',
    '--task_name',
    help=('\n'.join(example_dict.keys())))

argparser.add_argument(
    '-p',
    '--param_list',
    help=('Any input params to pass to the test you run'),
    nargs='*'
)

args = argparser.parse_args()

'''  initialize Kneron USB device '''
kdp_wrapper.init_log("/tmp/", "mzt.log")

if kdp_wrapper.lib_init() < 0:
    print("init for kdp host lib failed.\n")
    exit(-1)

print("adding devices....\n")

category = "KL520"
if category == "KL520":
    pid = KL520_PRODUCT_ID
else:
    print("Task category not in KL520: ", category)
    exit(-1)

dev_idx = -1

# dev_list is the list of [scan_index, isConnectable, vendor_id, product_id, link_speed, serial_number, device_path]
ret, dev_list = kdp_wrapper.scan_usb_devices()

for i in range(len(dev_list)):
    if (dev_list[i][3] == pid):#get the first KL520 or KL720 device
        # found it, now try to connect
        dev_idx = kdp_wrapper.connect_usb_device(dev_list[i][0])
        break

# print("dev_idx:", dev_idx)

if dev_idx < 0:
    print("add device failed.\n")
    exit(-1)

#kdp_wrapper.reset_sys(dev_idx, 0x10000206)

print("start kdp host lib....\n")
if kdp_wrapper.lib_start() < 0:
    print("start kdp host lib failed.\n")
    exit(-1)

user_id = 0



'''  de-initialize Kneron USB device '''
print("de init kdp host lib....\n")
# image input resolution, for HW preprocess, image size should be larger than model input size
image_source_w = 640
image_source_h = 480


# model information
model_file = "../input_models/KL520/yolo_face_nose.nef"
model_id = constants.ModelType.CUSTOMER.value


model_input_col = 480
model_input_row = 256

# the parameters for postprocess
anchor_path = './common/pre_post_process/yolo/models/anchors_v3.txt'
class_path = './common/class_lists/face_class'


# crop parameter
crop_top = 0
crop_bottom = 16
crop_left = 0
crop_right = 0


def end_det(dev_idx):
    """Ends DME mode for the specified device."""
    api.kdp_end_dme(dev_idx)
global nose_acne, nose_scar, l_nf, r_nf, l_acne, r_acne, l_scar, r_scar
nose_acne = 0
nose_scar = 0
l_nf = 0
r_nf = 0
l_acne = 0
r_acne = 0
l_scar = 0
r_scar = 0
def post_handler(dev_idx, raw_res, captured_frames_path):
    global time_path
    global nose_acne, nose_scar, l_nf, r_nf, l_acne, r_acne, l_scar, r_scar
    captured_frames = captured_frames_path

    # the parameters for postprocess
    model_input_shape = (model_input_col, model_input_row)
    score_thres       = 0.17
    nms_thres         = 0.5
    keep_aspect_ratio = True
    original_height =  image_source_h - crop_top - crop_bottom
    original_width = image_source_w - crop_left - crop_right
    is_v5 = True
    sigmoid_in_post_v5 = True


    raw_res.sort(key=lambda x: x.size)

    det_res = yolo_postprocess_(raw_res, anchor_path, class_path, original_height, original_width, model_input_shape,
                              score_thres, nms_thres, keep_aspect_ratio, is_v5, sigmoid_in_post_v5)

    #dlib dace detector
    '''detector = dlib.get_frontal_face_detector()
    detect, _, _ = detector.run(captured_frames[0], 0,-0.2)
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    for i, d_point in enumerate(detect):
        x1 = d_point.left()
        y1 = d_point.top()
        x2 = d_point.right()
        y2 = d_point.bottom()'''
    kneron_predict_result, dets = kdp_wrapper.draw_capture_result(dev_idx, det_res, captured_frames, "yolo", xywh=False)
    #print(detect)
    with open('./common/class_lists/face_class') as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    if dets:
        if dets[0][5] == 0:
            tmp = 0
            x1_n = dets[0][0]
            y1_n = dets[0][1]
            x2_n = dets[0][2]
            y2_n = dets[0][3]
            nose_center_x = (int)((x1_n + x2_n)/2)
            nose_center_y = (int)((y1_n + y2_n)/2)
            for det in dets:
                if tmp == 0:
                    tmp += 1
                else:
                    x1 = det[0]
                    y1 = det[1]
                    x2 = det[2]
                    y2 = det[3]
                    class_num = det[5]
                    d_center_x = (int)((x1 + x2)/2)
                    d_center_y = (int)((y1 + y2)/2)
                    if x1_n < d_center_x and x2_n > d_center_x and y1_n < d_center_y and y2_n > d_center_y and class_num == 4 and nose_acne == 0:
                        print('nose has ' + 'ance')
                        nose_acne = 1
                        a = detect_var.get()
                        detect_var.set(a + 'nose:' + 'ance' + '\n')
                        cv2.imwrite(time_path + '/nose_ance.png' ,kneron_predict_result)
                    elif x1_n < d_center_x and x2_n > d_center_x and y1_n < d_center_y and y2_n > d_center_y and class_num == 3 and nose_scar == 0:
                        nose_scar = 1
                        a = detect_var.get()
                        detect_var.set(a + 'nose:' + 'scar' + '\n')
                        cv2.imwrite(time_path + '/nose_scar.png' ,kneron_predict_result)
                    else:
                        if nose_center_x < d_center_x and class_num == 3 and r_scar == 0:
                            r_scar = 1
                            print('right face has ' + class_names[class_num])
                            a = detect_var.get()
                            detect_var.set(a + 'right face:' + 'scar' + '\n')
                            cv2.imwrite(time_path + '/r_scar.png' ,kneron_predict_result)
                        elif nose_center_x < d_center_x and class_num == 4 and r_acne == 0:
                            r_acne = 1
                            print('right face has ' + class_names[class_num])
                            a = detect_var.get()
                            detect_var.set(a + 'right face:' + 'acne' + '\n')
                            cv2.imwrite(time_path + '/r_acne.png' ,kneron_predict_result)
                        elif nose_center_x < d_center_x and class_num == 2 and r_nf == 0:
                            r_nf = 1
                            print('right face has ' + class_names[class_num])
                            a = detect_var.get()
                            detect_var.set(a + 'right face:' + 'nf  ' + '\n')
                            cv2.imwrite(time_path + '/r_nf.png' ,kneron_predict_result)
                        elif nose_center_x > d_center_x and class_num == 3 and l_scar == 0: 
                            l_scar = 1
                            print('left face has ' + class_names[class_num])
                            a = detect_var.get()
                            detect_var.set(a + 'left face:' + 'scar' + '\n')
                            cv2.imwrite(time_path + '/l_scar.png' ,kneron_predict_result)
                        elif nose_center_x > d_center_x and class_num == 4 and l_acne == 0: 
                            l_acne = 1
                            print('left face has ' + class_names[class_num])
                            a = detect_var.get()
                            detect_var.set(a + 'left face:' + 'acne' + '\n')
                            cv2.imwrite(time_path + '/l_acne.png' ,kneron_predict_result)
                        elif nose_center_x > d_center_x and class_num == 2 and l_nf == 0: 
                            l_nf = 1
                            print('left face has ' + class_names[class_num])
                            a = detect_var.get()
                            detect_var.set(a + 'left face:' + 'nf  ' + '\n')
                            cv2.imwrite(time_path + '/l_nf.png' ,kneron_predict_result)
    #print(dets)
    #cv2.rectangle(kneron_predict_result, ((int)((x1+x2)/2), (int)((y1+y2)/2)), ((int)((x1+x2)/2), (int)((y1+y2)/2)), (0, 255, 0), 4, cv2.LINE_AA)
    img = cv2.resize(kneron_predict_result, (704,528))
    cv2image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
    current_image = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=current_image)
    camera.imgtk = imgtk
    camera.config(image=imgtk)
    root.update()
def user_test_single_dme(dev_idx, loop):
    """Test single DME."""
    # As sigmoid is not supported by KL520 fw, the sigmoid nodes have been cut,
    # and NEF does not include sigmoid nodes.
    image_cfg = (constants.IMAGE_FORMAT_RAW_OUTPUT | constants.NPU_FORMAT_RGB565 | constants.IMAGE_FORMAT_SUB128)

    #for model 480x256, crop image 640x480 into 640x464 to keep the pad value smaller than 127
    crop_box=(crop_top, image_source_h-crop_bottom, crop_left, image_source_w-crop_right)

    # Load model into Kneron device.
    dme_config = kdp_wrapper.init_dme_config(model_id, 3, image_cfg, image_col=image_source_w, image_row=image_source_h,image_ch = 2,\
    crop_box=crop_box)

    ret = kdp_wrapper.dme_load_model(dev_idx, model_file, dme_config)
    if ret == -1:
        return -1

    app_id = 0 # if app_id is 0, output raw data for kdp_wrapper.kdp_dme_inference

    # Setup video capture device.
    image_size = image_source_w * image_source_h * 2 # rgb565
    frames = []
    capture = kdp_wrapper.setup_capture(0, image_source_w, image_source_h)
    if capture is None:
        return -1

    # Send 1 image to the DME image buffers.
    ssid = kdp_wrapper.dme_fill_buffer(dev_idx, capture, image_size, frames, image_source_w, image_source_h)
    if ssid == -1:
        print("dme_fill_buffer fail!")
        return -1

    return kdp_wrapper.dme_pipeline_inference(
        dev_idx, app_id, loop, image_size, capture, ssid, frames, post_handler, image_source_w, image_source_h)

def user_test(dev_idx, _user_id):
    # dme test
    ret = user_test_single_dme(dev_idx, 0)
    kdp_wrapper.end_det(dev_idx)
    return ret
def cam_predict():
    user_test(0,0)
    
def make_dir(path):
    os.mkdir(path)
def show_notice():
    print('Notice')
    global nose_acne, nose_scar, l_nf, r_nf, l_acne, r_acne, l_scar, r_scar
    if nose_acne or nose_scar or l_acne or r_acne or l_scar or r_scar:
        notice_var.set('建議選擇溫和不刺激的洗面乳，\n洗臉的次數一天也以不超過兩次為佳。')
    else:
        notice_var.set('臉部膚質正常')
    det_end = Image.open("./image/func1/det_end.png")
    det_end = det_end.resize((704,528),Image.ANTIALIAS)
    det_end_tk =ImageTk.PhotoImage(det_end)
    camera.imgtk = det_end_tk
    camera.config(image=det_end_tk)
    root.update()
def show_page1():
    article_ori = Image.open("./image/article/home_title.png")
    article_re= article_ori.resize((180,674),Image.ANTIALIAS)
    article_tk=ImageTk.PhotoImage(article_re)
    article = tk.Canvas(root, width=180, height=674, background='#C19892', highlightthickness=0)
    article.grid()
    article.create_image(2, 2, image=article_tk, anchor=tk.NW)
    article.place(x=-28,y=0)
def show_func1_page1():
    global time_path
    func1_article.place(x=-28,y=0)
    func1_cap_btn.place(x=530, y=580)
    func1_back_btn.place(x=350, y=580)
    func1_detect_label.place(x = 880, y = 30)
    func1_tip_label.place(x = 880, y = 360)
    camera.place(x=200,y=40)
    detect_word.place(x = 900, y = 110)
    notice_word.place(x = 900, y = 440)
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    make_dir('./detect_result_save/' + current_time)
    time_path = './detect_result_save/' + current_time
def del_func1_page1():
    func1_article.place_forget(),func1_cap_btn.place_forget(),func1_back_btn.place_forget()
    camera.place_forget(),func1_detect_label.place_forget(),func1_tip_label.place_forget(),detect_word.place_forget()
    notice_word.place_forget()
    notice_var.set('')
    detect_var.set('')
    global nose_acne, nose_scar, l_nf, r_nf, l_acne, r_acne, l_scar, r_scar
    nose_acne = 0
    nose_scar = 0
    l_nf = 0
    r_nf = 0
    l_acne = 0
    r_acne = 0
    l_scar = 0
    r_scar = 0
def end_predict():
    end_det(0)
def show_func1_page2():
    func1_article.place(x=-28,y=0)
    result_txt1_label.place(x=240,y=50)
    result_txt2_label.place(x=190,y=430)
    func1_tip_label.place(x=570,y=70)
    func1_back_pg2_btn.place(x=635,y=490)
    func1_check_pg2_btn.place(x=815,y=490)
def del_func1_page2():
    func1_article.place_forget(),result_txt1_label.place_forget(),result_txt2_label.place_forget()
    func1_tip_label.place_forget(),func1_back_pg2_btn.place_forget(),func1_check_pg2_btn.place_forget()
def del_main():
    article.place(x=-500,y=0)
    page1.place_forget(),page2.place_forget(),page3.place_forget()
    home_item_label.place_forget()
def main_page():
    func1_article.place(x=-500,y=0)
    article.place(x=-28,y=0)
    page1.place(x=460, y=80)
    page2.place(x=460, y=250)
    page3.place(x=460, y=420)
    home_item_label.place(x=1050, y=570)
    
root=tk.Tk()
root.title("AI痘膚頭")
root.configure(background='#FFFFFF')
root.attributes('-zoomed',True)
FontStyle = tkFont.Font(family='',size=14)
#home title
article_ori = Image.open("./image/article/title.png")
article_re= article_ori.resize((180,690),Image.ANTIALIAS)
article_tk=ImageTk.PhotoImage(article_re)
article = tk.Canvas(root, width=180, height=690, background='#C19892', highlightthickness=0)
article.grid()
article.create_image(2, 2, image=article_tk, anchor=tk.NW)
article.place(x=-28,y=0)
#home button
func1 = Image.open("./image/home/function1.png")
func1= func1.resize((408,120),Image.ANTIALIAS)
func1_tk=ImageTk.PhotoImage(func1)
func2 = Image.open("./image/home/function2.png")
func2= func2.resize((408,120),Image.ANTIALIAS)
func2_tk=ImageTk.PhotoImage(func2)
func3 = Image.open("./image/home/function3.png")
func3= func3.resize((408,120),Image.ANTIALIAS)
func3_tk=ImageTk.PhotoImage(func3)
page1 = tk.Button(root,bg="#FFFFFF",activebackground="#FFFFFF",bd=0,highlightthickness = 0, image=func1_tk,relief= 'flat', command=lambda:[del_main(),show_func1_page1(),cam_predict()])
page2 = tk.Button(root,bg="#FFFFFF",activebackground="#FFFFFF",bd=0,highlightthickness = 0, image=func2_tk,relief= 'flat', command=lambda:[close_window(),shopping()])
page3 = tk.Button(root,bg="#FFFFFF",activebackground="#FFFFFF",bd=0,highlightthickness = 0, image=func3_tk,relief= 'flat', command=lambda:[close_window(),cosmetic()])
#home item
home_item = Image.open("./image/home/home_item.png")
home_item= home_item.resize((224,112),Image.ANTIALIAS)
home_item_tk=ImageTk.PhotoImage(home_item)
home_item_label = tk.Label(root,image = home_item_tk,bg = '#FFFFFF')

#func1 title
func1_article_ori = Image.open("./image/article/func1_title.png")
func1_article_re= func1_article_ori.resize((180,690),Image.ANTIALIAS)
func1_article_tk=ImageTk.PhotoImage(func1_article_re)
func1_article = tk.Canvas(root, width=180, height=690, background='#C19892', highlightthickness=0)
func1_article.grid()
func1_article.create_image(2, 2, image=func1_article_tk, anchor=tk.NW)
#func1 check back
func1_cap = Image.open("./image/check.png")
func1_cap= func1_cap.resize((160,61),Image.ANTIALIAS)
func1_cap_tk=ImageTk.PhotoImage(func1_cap)
func1_cap_btn = tk.Button(root,bg="#FFFFFF",activebackground="#FFFFFF",bd=0,highlightthickness = 0, image=func1_cap_tk,relief= 'flat', command=lambda:[end_predict(),show_notice()])
func1_back = Image.open("./image/back.png")
func1_back= func1_back.resize((160,61),Image.ANTIALIAS)
func1_back_tk=ImageTk.PhotoImage(func1_back)
func1_back_btn = tk.Button(root,bg="#FFFFFF",activebackground="#FFFFFF",bd=0,highlightthickness = 0, image=func1_back_tk,relief= 'flat', command=lambda:[del_func1_page1(),end_predict(),main_page()])
func1_back_pg2 = Image.open("./image/back.png")
func1_back_pg2= func1_back_pg2.resize((160,61),Image.ANTIALIAS)
func1_back_pg2_tk=ImageTk.PhotoImage(func1_back_pg2)
func1_back_pg2_btn = tk.Button(root,bg="#FFFFFF",activebackground="#FFFFFF",bd=0,highlightthickness = 0, image=func1_back_tk,relief= 'flat', command=lambda:[del_func1_page1(),main_page()])
func1_check_pg2 = Image.open("./image/check.png")
func1_check_pg2= func1_check_pg2.resize((160,61),Image.ANTIALIAS)
func1_check_pg2_tk=ImageTk.PhotoImage(func1_check_pg2)
func1_check_pg2_btn = tk.Button(root,bg="#FFFFFF",activebackground="#FFFFFF",bd=0,highlightthickness = 0, image=func1_check_pg2_tk,relief= 'flat', command=lambda:[])

#camera
camera = tk.Label(root,height=500,width=650,bg="#FFFFFF")  # initialize image camera2
#func1 item
result_txt1 = Image.open("./image/func1/result1.png")
result_txt1= result_txt1.resize((210,62),Image.ANTIALIAS)
result_txt1_tk=ImageTk.PhotoImage(result_txt1)
result_txt1_label = tk.Label(root,image = result_txt1_tk,bg = '#FFFFFF')

func1_detect = Image.open("./image/func1/detect_result.png")
func1_detect = func1_detect.resize((368,275),Image.ANTIALIAS)
func1_detect_tk=ImageTk.PhotoImage(func1_detect)
func1_detect_label = tk.Label(root,image = func1_detect_tk,bg = '#FFFFFF')

func1_tip = Image.open("./image/func1/tips.png")
func1_tip= func1_tip.resize((368,275),Image.ANTIALIAS)
func1_tip_tk=ImageTk.PhotoImage(func1_tip)
func1_tip_label = tk.Label(root,image = func1_tip_tk,bg = '#FFFFFF')
#tkvar
detect_var = tk.StringVar()
detect_var.set('')
notice_var = tk.StringVar()
notice_var.set('')
#func1_text_label
detect_word = tk.Label(root, textvariable = detect_var, font = FontStyle, bg="#D8D6D7")
notice_word = tk.Label(root, textvariable = notice_var, font = FontStyle, bg="#D8D6D7")
main_page()
root.mainloop()

