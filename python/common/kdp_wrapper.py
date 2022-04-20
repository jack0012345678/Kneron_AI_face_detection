"""
This is KDP wrapper.
"""
from __future__ import absolute_import
import ctypes
import math
import struct
import sys
from time import sleep

import cv2
import numpy as np

from common import constants
import kdp_host_api as api

def pad_up_16(value):
    """Aligns value argument to 16."""
    return math.ceil(value / 16) * 16

def sigmoid(x):
    """
    sigmoid for numpy array
    """
    return 1 / (1 + np.exp(-x))

def init_log(directory, name):
    """Initialize the host lib internal log.

    Returns 0 on success and -1 on failure.

    Arguments:
        directory: String name of directory
        name: String name of log file
    """
    return api.kdp_init_log(directory.encode(), name.encode())

def lib_init():
    """Initialize the host library.

    Returns 0 on success and -1 on failure.
    """
    return api.kdp_lib_init()

def scan_usb_devices():
    """Scan all Kneron devices and report a list.

    Returns 0. Will also return list of all available devices.
    """
    # KDPDeviceInfoList instance
    dev_info_list = constants.KDPDeviceInfoList()

    # Create double pointer to KDPDeviceInfoList
    r_data = ctypes.pointer(ctypes.pointer(dev_info_list))
    ret = api.kdp_scan_usb_devices(r_data)

    # Get the device info list
    dev_info_list = []

    dev_list = r_data.contents.contents
    for i in range(dev_list.num_dev):
        dev_info = []
        dev_info.append(dev_list.kdevice[i].scan_index)
        dev_info.append(dev_list.kdevice[i].isConnectable)
        dev_info.append(dev_list.kdevice[i].vendor_id)
        dev_info.append(dev_list.kdevice[i].product_id)
        dev_info.append(dev_list.kdevice[i].link_speed)
        dev_info.append(dev_list.kdevice[i].serial_number)
        dev_info.append(dev_list.kdevice[i].device_path.decode())
        dev_info_list.append(dev_info)

    # print("return list:", dev_info_list)
    return ret, dev_info_list

def connect_usb_device(scan_index=1):
    """Connect to a Kneron device via 'scan_index'.

    Returns device index on success and negative value on failure.

    Arguments:
        scan_index: Integer device index to connect
    """
    return api.kdp_connect_usb_device(scan_index)

def lib_start():
    """Start the host library to wait for messages.

    Returns 0 on success and -1 on failure.
    """
    return api.kdp_lib_start()

def lib_de_init():
    """Free the resources used by host lib.

    Returns 0 on success and -1 on failure.
    """
    return api.kdp_lib_de_init()

def reset_sys(device_index, reset_mode):
    """System reset.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        reset_mode: Integer mode
            0 - no operation
            1 - reset message protocol
            3 - switch to suspend mode
            4 - switch to active mode
            255 - reset whole system
            256 - system shutdown (RTC)
            0x1000xxxx - reset debug output level
    """
    ret = api.kdp_reset_sys(device_index, reset_mode)

    if ret:
        print("Could not reset sys...")
    else:
        print("Sys reset mode succeeded...")

    return ret

def update_fw(device_index, module_id, fw_file, fw_size):
    """Update firmware.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        module_id: Integer module ID of which firmware to be updated
            0 - no operation
            1 - SCPU module
            2 - NCPU module
        fw_file: String path to firmware data
        fw_size: Integer size of firmware data
    """
    img_buf, img_buf_size = read_file_to_buf_with_size(fw_file, fw_size)
    if img_buf_size <= 0:
        print(f"Reading model file, {fw_file}, failed: {img_buf_size}")
        return img_buf_size

    module = ctypes.c_uint32(module_id)
    ret = api.kdp_update_fw(device_index, ctypes.byref(module), img_buf, img_buf_size)

    module = {0: "nothing", 1: "SCPU", 2: "NCPU"}
    module_string = module.get(module_id, "UNKNOWN MODULE")
    if ret:
        print(f"Could not update {module_string}...")
    else:
        print(f"Update {module_string} firmware succeeded...")

    return ret

def report_sys_status(device_index):
    """Reports device status.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
    """
    sfw_id = ctypes.c_uint32(0)
    sbuild_id = ctypes.c_uint32(0)
    sys_status = ctypes.c_uint16(0)
    app_status = ctypes.c_uint16(0)
    nfw_id = ctypes.c_uint32(0)
    nbuild_id = ctypes.c_uint32(0)

    ret = api.kdp_report_sys_status(
        device_index, ctypes.byref(sfw_id), ctypes.byref(sbuild_id), ctypes.byref(sys_status),
        ctypes.byref(app_status), ctypes.byref(nfw_id), ctypes.byref(nbuild_id))

    if ret:
        print("Could not report sys status...")
        return ret
 
    print("Report sys status succeeded...")
    architecture = sfw_id.value >> 24
    major = (sfw_id.value & 0x00ff0000) >> 16
    minor = (sfw_id.value & 0x0000ff00) >> 8
    update = (sfw_id.value & 0x000000ff)
    build = sbuild_id.value

    print(f"\nFW firmware_id {architecture}.{major}.{minor}.{update}, build_id {build}\n")

    return ret

def init_dme_config(model_id, output_num, image_format, image_col=constants.IMAGE_SOURCE_W_DEFAULT,
                    image_row=constants.IMAGE_SOURCE_H_DEFAULT, image_ch=3, ext_param=[0.0], 
                    crop_box = (0,constants.IMAGE_SOURCE_H_DEFAULT,0,constants.IMAGE_SOURCE_W_DEFAULT)):
    """Initialize DME config instance for configuration.

    Returns initialized KDPDMEConfig instance.

    Arguments:
        model_id: Integer model ID to be inferenced
        output_num: Integer number of model outputs
        image_format: Integer format of input image
        image_col: Integer width of input image
        image_row: Integer height of input image
        image_ch: Integer number of channels in input image
        ext_param: List of postprocess parameters
    """
    # print("model_id, output_num, image_col, image_row, image_ch, image_format",\
    #     model_id, output_num,  image_col, image_row, image_ch, image_format)
    return constants.KDPDMEConfig(
        model_id=model_id, output_num=output_num, image_col=image_col, image_row=image_row,
        image_ch=image_ch, image_format=image_format, crop_box=crop_box,ext_param=ext_param)

def dme_load_model(device_index, model_file, dme_config):
    """Loads the model and sets DME configurations.

    Returns 0 on success and error code on failure.

    Arguments:
        device_index: Integer connected device ID
        model_file: String path to model
        dme_config: KDPDMEConfig instance
    """
    print("Loading models to Kneron Device: ")

    p_buf, n_len = read_file_to_buf_with_size(model_file, constants.MAX_MODEL_SIZE_720)
    if p_buf is None:
        return -1

    print("\nStarting DME mode...")

    ret_size = ctypes.c_uint32(0)

    ret = api.kdp_start_dme_ext(device_index, p_buf, n_len, ctypes.byref(ret_size))
    if ret:
        print(f"Could not set to DME mode: {ret_size}...")
        return ret

    print("\nDME mode succeeded...")
    print("Model loading successful")
    sleep(constants.SLEEP_TIME)

    print("dme_config:",dme_config)

    ret = dme_configure(device_index, dme_config)
    if ret:
        return ret

    return 0

def read_file_to_buf_with_size(image_file, image_size):
    """Reads input image into a buffer.

    Returns the image buffer and length of the input image. Imabe buffer will be none,
    and length will be negative on failure.

    Arguments:
        image_file: String path to the input image
        image_size: Integer size of the input image
    """
    buffer = (ctypes.c_char * image_size)()
    length = api.read_file_to_buf(buffer, image_file.encode(), image_size)
    if length <= 0:
        print(f"Reading file {image_file} failed: {length}...")
        return None, length
    return buffer, length

def dme_configure(device_index, dme_config, to_print=True):
    """Change the DME configurations.

    Arguments:
        device_index: Integer connected device ID
        dme_config: KDPDMEConfig instance
        to_print: Flag to enable prints
    """
    if to_print:
        print("\nStarting DME configure...")

    dme_config_p = ctypes.cast(ctypes.pointer(dme_config), ctypes.POINTER(ctypes.c_char))
    ret_model = ctypes.c_uint32(0)
    ret = api.kdp_dme_configure(
        device_index, dme_config_p, dme_config.struct_size(), ctypes.byref(ret_model))

    if ret:
        print(f"Could not set to DME configure mode: {ret_model.value}...")
        return ret

    if to_print:
        print(f"DME configure model [{ret_model.value}] succeeded...\n")
    sleep(constants.SLEEP_TIME)

    return 0


def dme_fill_buffer(device_index, capture, input_size, frames, width=640, height=480, pre_handler=None):
    """Send 1 image to the DME image buffers using the capture device.

    Returns session ID.

    Arguments:
        device_index: Integer connected device ID
        capture: Active cv2 video capture instance
        input_size: Integer size of the input image
        frames: List of frames captured by the video capture instance
        pre_handler: Function to perform preprocessing; None uses capture, otherwise use frames
    """
    print("Starting DME inference...")
    inf_res = (ctypes.c_char * 0x300000)()
    res_flag = ctypes.c_bool(False)
    mode = 1
    model_id = 0
    ssid = ctypes.c_uint32(0)

    if capture is not None:
        img_buf = capture_frame(capture, frames, width, height)
    elif pre_handler is not None:
        data = pre_handler(frames[0])
        img_buf = convert_numpy_to_char_p(data, size=input_size)
    else:
        print("Both capture and pre_handler function with input images in frames"
              " cannot be None...")
        return -1

    ret = api.kdp_dme_inference(
        device_index, img_buf, input_size, ctypes.byref(ssid), ctypes.byref(res_flag), inf_res, mode, model_id)
    if ret:
        print(f"DME inference failed: {ret}...")
        return -1

    return ssid.value

def convert_float_to_rgba(data, radix, platform, set_hwc=False):
    """Converts the NumPy float data into RGBA.

    Arguments:
        data: Input NumPy data
        radix: Radix of the input node
        platform: Integer platform (520 or 720)
        set_hwc: Flag to indicate if transpose is needed
    """
    if len(data.shape) == 3:    # add batch dimension
        data = np.reshape(data, (1, *data.shape))

    if set_hwc:                 # bchw -> bhwc
        data = np.transpose(data, (0, 2, 3, 1))
    _, height, width, channel = data.shape

    if platform == 520:
        width_aligned = 16 * math.ceil(width / 16.0)
    else:
        width_aligned = 4 * math.ceil(width / 4.0)

    aligned_data = np.zeros((1, height, width_aligned, 4))
    aligned_data[:, :height, :width, :channel] = data
    aligned_data *= 2 ** radix
    aligned_data = aligned_data.astype(np.int8)
    return aligned_data

def convert_numpy_to_char_p(frame, color=None, size=constants.IMAGE_SIZE_RGB565_DEFAULT):
    """Converts NumPy array into ctypes char pointer.

    Arguments:
        frame: NumPy array from image
        color: Integer indicating color conversion
        size: Integer size of the frame array
    """
    new_frame = frame
    if color is not None:
        new_frame = cv2.cvtColor(frame, color)
    new_frame = new_frame.reshape(size)
    new_frame = new_frame.astype(np.uint8)
    return new_frame.ctypes.data_as(ctypes.POINTER(ctypes.c_char))

def dme_pipeline_inference(device_index, app_id, loops, input_size, capture,
                           prev_ssid, frames, post_handler, width=640, height=480, pre_handler=None):
    """Send the rest of images and get the results.

    Arguments:
        device_index: Integer connected device ID
        app_id: Integer ID of application to be run
        loops: Integer number of images to get results
        input_size: Integer size of the input image
        capture: Active cv2 video capture instance
        prev_ssid: Integer previous session ID, should be returned from dme_fill_buffer()
        frames: List of frames captured by the video capture instance
        post_handler: Function to process the results of the inference
        pre_handler: Function to perform preprocessing
    """
    inf_res = (ctypes.c_char * 0x300000)()
    res_flag = ctypes.c_bool(False)
    mode = 1
    model_id = 0
    ssid = ctypes.c_uint32(0)
    inf_size = ctypes.c_uint32(0)

    index = 1
    num_images = len(frames)

    keep_loop = 0
    if loops==0: # while forever
        keep_loop = 1
        loops = 1
    print("pre_handler:",pre_handler)
    while loops:
        prev_ssid = ctypes.c_uint16(prev_ssid)

        if capture is not None:
            img_buf = capture_frame(capture, frames, width, height)
        elif pre_handler is not None:
            prev_index = (index - 1) % num_images
            cur_index = index % num_images
            data = pre_handler(frames[cur_index])
            img_buf = convert_numpy_to_char_p(data, size=input_size)
        else:
            print("Both capture and pre_handler function with input images in frames"
                  " cannot be None...")
            return -1

        ret = api.kdp_dme_inference(device_index, img_buf, input_size, ctypes.byref(ssid),
                                    ctypes.byref(res_flag), inf_res, mode, model_id)
        if ret:
            print(f"DME inference failed: {ret}...")
            return -1

        # get status for previous session
        while 1:
            status = ctypes.c_uint16(0)  # Must re-initialize status to 0
            ret = api.kdp_dme_get_status(device_index, ctypes.byref(prev_ssid),
                                         ctypes.byref(status), ctypes.byref(inf_size), inf_res)
            if ret:
                print(f"Get DME status failed: {ret}...")
                return -1

            if status.value == 1:
                # print("inf_size = ",inf_size.value)
                npraw_data = dme_get_result(device_index, inf_size.value, app_id)
                # print("len npraw_data",len(npraw_data))
                if capture is not None:
                    post_handler(device_index, npraw_data, frames)
                else:
                    post_handler(device_index, npraw_data, frames[prev_index])
                break

        prev_ssid = ssid.value

        if (keep_loop == 0):
            loops -= 1
        index += 1

    # Get last 1 results
    prev_ssid = ctypes.c_uint16(prev_ssid)
    while 1:
        status = ctypes.c_uint16(0)  # Must re-initialize status to 0
        ret = api.kdp_dme_get_status(device_index, ctypes.byref(prev_ssid),
                                     ctypes.byref(status), ctypes.byref(inf_size), inf_res)
        if ret:
                print(f"Get DME status failed: {ret}...")
                return -1

        if status.value == 1:
            npraw_data = dme_get_result(device_index, inf_size.value, app_id)
            if capture is not None:
                post_handler(device_index, npraw_data, frames)
            else:
                prev_index = (index - 1) % num_images
                post_handler(device_index, npraw_data, frames[prev_index])
            break

    return 0

def dme_get_result(dev_idx, inf_size, app_id):
    """Gets inference results.

    Arguments:
        dev_idx: Integer connected device ID
        inf_size: Integer inference data size
        app_id: Integer ID of application to be run
    """
    inf_res = (ctypes.c_char * inf_size)()
    ret = api.kdp_dme_retrieve_res(dev_idx, 0, inf_size, inf_res)
    if ret:
        print("DME get result failed: {}\n".format(ret))
        return -1

    if (app_id == 0): # raw output
        # Prepare for postprocessing
        listdata = [ord(byte) for byte in inf_res]
        npdata = np.asarray(listdata)

        fp_header_res = ctypes.cast(
            ctypes.byref(inf_res), ctypes.POINTER(constants.RawFixpointData)).contents
        output_num = fp_header_res.output_num

        outnode_params_res = ctypes.cast(
            ctypes.byref(fp_header_res.out_node_params),
            ctypes.POINTER(constants.OutputNodeParams * output_num)).contents

        npraw_data_array = []
        data_offset = 0
        for param in outnode_params_res:
            height = param.height
            channel = param.channel
            width = param.width
            radix = param.radix
            scale = param.scale

            # print(output_num, height, channel, width, pad_up_16(width), radix, scale)

            # offset in bytes for TOTAL_OUT_NUMBER + (H/C/W/RADIX/SCALE) + (H/C/W/RADIX/SCALE)
            offset = ctypes.sizeof(ctypes.c_int) + output_num * ctypes.sizeof(constants.OutputNodeParams)
            # print("offset ", offset, ctypes.sizeof(c_int), ctypes.sizeof(OutputNodeParams))

            # get the fixed-point data
            npdata = npdata.astype("int8")

            raw_data = npdata[offset + data_offset:offset + data_offset + height * channel * pad_up_16(width)]
            data_offset += height * channel * pad_up_16(width)
            # print(raw_data.shape, offset, offset + height*channel*pad_up_16(width), height*channel*pad_up_16(width))
            raw_data = raw_data.reshape(height, channel, pad_up_16(width))
            raw_data = raw_data[:, :, :width]

            # save the fp data into numpy array and convert to float
            npraw_data = np.array(raw_data)
            npraw_data = npraw_data.transpose(0, 2, 1) / (2 ** radix) / scale
            npraw_data_array.append(npraw_data)

        return npraw_data_array
    elif app_id == constants.AppID.APP_AGE_GENDER: # age_gender
        result = cast_and_get(inf_res, constants.FDAgeGenderRes)
        det_res = []
        FACE_SCORE_THRESHOLD = 0.8
        if result.fd_res.score > FACE_SCORE_THRESHOLD:
            # print("[INFO] FACE DETECT (x1, y1, x2, y2, score) = {}, {}, {}, {}, {}\n".format(
            #     result.fd_res.x1, result.fd_res.y1, result.fd_res.x2, result.fd_res.y2,
            #     result.fd_res.score))
            if not result.ag_res.age and not result.ag_res.ismale:
                # print("[INFO] FACE TOO SMALL\n")
                res = [int(result.fd_res.x1), int(result.fd_res.y1), int(result.fd_res.x2), int(result.fd_res.y2),
                       float(result.fd_res.score), 0, 3]  # age:0 gender:3
            else:
                # gender = "Male" if result.ag_res.ismale else "Female"
                # print("[INFO] AGE_GENDER (Age, Gender) = {}, {}\n".format(
                #     result.ag_res.age, gender))
                res = [int(result.fd_res.x1), int(result.fd_res.y1), int(result.fd_res.x2), int(result.fd_res.y2),
                       float(result.fd_res.score), int(result.ag_res.age),
                       int(result.ag_res.ismale)]
            det_res.append(res)
        return det_res
    else: # od, yolo
        od_header_res = cast_and_get(inf_res, constants.ObjectDetectionRes)
        det_res = []

        r_size = 4
        if r_size >= 4:
            box_result = cast_and_get(od_header_res.boxes, constants.BoundingBox * od_header_res.box_count)

            for box in box_result:
                x1 = int(box.x1)
                y1 = int(box.y1)
                x2 = int(box.x2)
                y2 = int(box.y2)
                score = float(box.score)
                class_num = int(box.class_num)
                res = [x1, y1, x2, y2, class_num, score]
                det_res.append(res)

        return det_res

def draw_capture_result(device_index, dets, frames, det_type,
                        xywh=False, apple_img=None, apple_list=None):
    """Draw the detection results on the given frames.

    Arguments:
        device_index: Integer connected device ID
        dets: List of detection results
        frames: List of frames, result will be drawn on first frame
        det_type: String indicating which model result dets corresponds to
        xywh: Flag indicating if dets are in xywh, default is x1y1x2y2
        apple_img: Apple image array to display
        apple_list: List of apples, only used if det_type is 'apple'
    """
    x1_0 = 0
    y1_0 = 0
    x2_0 = 0
    y2_0 = 0
    # score_0 = 0

    # for multiple faces
    for det in dets:
        x1 = det[0]
        y1 = det[1]
        x2 = det[2]
        y2 = det[3]
        if det[5] == 0 or det[5] == 5 or det[5] == 6:
            face_center_x = (int)((x1 + x2)/2)
            face_center_y = (int)((y1 + y2)/2)
        else:
            face_center_x = 0
            face_center_y = 0           
        if xywh:    # convert xywh to x1y1x2y2
            x2 += x1
            y2 += y1

        if det_type == "age_gender":
            score = det[4]
            age = det[5]
            gender = det[6]
            if gender == 0:
                frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                frames[0] = cv2.putText(frames[0], str(age), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 0, 255), 2, cv2.LINE_AA)
            elif gender == 1:
                frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)
                frames[0] = cv2.putText(frames[0], str(age), (x1, y1), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (255, 0, 0), 2, cv2.LINE_AA)
        elif det_type == "yolo" or det_type == "fd_no_overlap" or det_type == "fd_mask_no_overlap":
            x1 = int(x1)
            y1 = int(y1)
            x2 = int(x2)
            y2 = int(y2)
            score = det[4]
            class_num = det[5]

            with open('./common/class_lists/face_class') as f:
                class_names = f.readlines()
            class_names = [c.strip() for c in class_names]
            if class_num:
                frames[0] = cv2.putText(frames[0], class_names[class_num], (x1,y1+25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)
                
            else:
                pass
            # frames[0] = cv2.putText(frames[0], str(class_num), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA) 
            if det_type == "fd_mask_no_overlap":
                if det[5] == 0 or det[5] == 5 or det[5] == 6:
                    if class_num == 2:
                        frames[0] = cv2.rectangle(frames[0], (face_center_x, face_center_y), (face_center_x, face_center_y), (0, 0, 255), 3)
                    elif class_num == 1:
                        frames[0] = cv2.rectangle(frames[0], (face_center_x, face_center_y), (face_center_x, face_center_y), (255, 0, 0), 3)
                else:
                    if class_num == 2:
                        frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                    elif class_num == 1:
                        frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)
            else:
                if det[5] == 0 or det[5] == 5 or det[5] == 6:
                    if class_num == 0:
                        frames[0] = cv2.rectangle(frames[0], (face_center_x, face_center_y), (face_center_x, face_center_y), (0, 0, 255), 3)
                    else:
                        frames[0] = cv2.rectangle(frames[0], (face_center_x, face_center_y), (face_center_x, face_center_y), (255, 0, 0), 3)
                else:
                    if class_num == 0:
                        frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                    else:
                        frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)  
        else:
            class_num = det[4]
            score = det[5]
            o_l = overlap(x1, y1, x2, y2, x1_0, y1_0, x2_0, y2_0)
            if o_l < 0.6:
                x1_0 = x1
                y1_0 = y1
                x2_0 = x2
                y2_0 = y2
                # score_0 = score
                if class_num == 2:
                    frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (0, 0, 255), 3)
                    #print("score of mask fd: ", score)
                elif class_num == 1:
                    frames[0] = cv2.rectangle(frames[0], (x1, y1), (x2, y2), (255, 0, 0), 3)
                    #print("score of fd: ", score)

                if det_type == "apple":
                    index = 0
                    remove_index = -1

                    # only remove one per frame
                    for apple in apple_list:
                        apple_x = apple[0] + 50
                        apple_y = apple[1] + 50

                        if x1 < apple_x < x2 and (y1 + (y2 - y1) / 2) < apple_y < y2:
                            remove_index = index
                        index += 1

                    if remove_index != -1:
                        del apple_list[remove_index]

    if det_type == "apple":
        #create transparent apple
        for apple in apple_list:
            result = np.zeros((100, 100, 3), np.uint8)
            apple_x = apple[0]
            apple_y = apple[1]
            bg = frames[0][apple_y:apple_y + apple_img.shape[0],
                           apple_x:apple_x + apple_img.shape[1]]

            alpha = apple_img[:, :, 3] / 255.0
            result[:, :, 0] = (1 - alpha) * bg[:, :, 0] + alpha * apple_img[:, :, 0]
            result[:, :, 1] = (1 - alpha) * bg[:, :, 1] + alpha * apple_img[:, :, 1]
            result[:, :, 2] = (1 - alpha) * bg[:, :, 2] + alpha * apple_img[:, :, 2]

            #added_image = cv2.addWeighted(bg, 0.4, apple_img, 0.3, 0)
            frames[0][apple_y:apple_y + apple_img.shape[0],
                      apple_x:apple_x + apple_img.shape[1]] = result
    #cv2.imshow("test", frames[0])
    a = frames[0]
    del frames[0]
    return a, dets


def end_det(dev_idx):
    """Ends DME mode for the specified device."""
    api.kdp_end_dme(dev_idx)

def setup_capture(cam_id, width, height):
    """Sets up the video capture device.

    Returns the video capture instance on success and None on failure.

    Arguments:
        cam_id: Integer camera ID
        width: Integer width of frames to capture
        height: Integer height of frames to capture
    """
    capture = cv2.VideoCapture(cam_id)
    if not capture.isOpened():
        print("Could not open video device!")
        return None
    if sys.platform != "darwin":
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    return capture



def capture_frame(cap, frames, width, height):
    """Frame read and convert to RGB565.

    Arguments:
        cap: Active cv2 video capture instance
        frames: List of frames for the video capture to add to
    """
    _cv_ret, frame = cap.read()

    if frame is None:
        print("fail to read from cam!")

    if sys.platform == "darwin":
        frame = cv2.resize(frame, (constants.IMAGE_SOURCE_W_DEFAULT, constants.IMAGE_SOURCE_H_DEFAULT),
                           interpolation=cv2.INTER_CUBIC)

    frame = cv2.flip(frame, 1)
    frames.append(frame)

    return convert_numpy_to_char_p(frame, cv2.COLOR_BGR2BGR565, width*height*2)



