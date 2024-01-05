# VITAL RING PPG_CONVERT_PROGRAM_V2.0.py by ZTACOM Co., Ltd.

# import warnings
# warnings.filterwarnings("ignore", category=RuntimeWarning)

import math
import os
import tkinter as tk
from tkinter import filedialog

import matplotlib
matplotlib.use('TkAgg')  # 'TkAgg'는 예시이며, 시스템에 맞는 다른 백엔드를 사용할 수도 있습니다.
import matplotlib.pyplot as plt

import numpy as np
from dotmap import DotMap
from scipy import signal
from scipy.signal import cheby2, filtfilt, find_peaks, convolve, butter, hilbert
from scipy.interpolate import CubicSpline

plt.rcParams['font.family'] = 'Malgun Gothic'  # 예: 'Malgun Gothic'을 사용할 경우
plt.rcParams['axes.unicode_minus'] = False  # 마이너스('-') 기호 깨짐 방지

import pandas as pd

app = None
show_graph = True

# ===============================================================================================
# yjs_reconstruction_ppg : 원본 PPG 데이터를 필터 처리하고, 신호 왜곡을 보정하고, 최정적으로 정규화된 파형으로 전처리함.
# ===============================================================================================
def yjs_reconstruction_ppg(filename, raw_ppg, order, lowcut, highcut, fs, btype='band'):
    global show_graph

    t = np.arange(len(raw_ppg)) / fs

    # 트렌드 제거 (하이패스 필터 사용)
    hp_cutoff = 0.15  # 하이패스 필터 컷오프 주파수 (Hz)
    b, a = butter(1, hp_cutoff / (0.5 * fs), btype='highpass')
    # print(f'트렌드 제거 (하이패스 필터 사용) time = {time}, b, a = {b}, {a}')
    signal_detrended = filtfilt(b, a, raw_ppg)

    # 1. PPG 신호에서 AC 및 DC 성분 추출
    b, a = butter(order, [lowcut / (fs / 2), highcut / (fs / 2)], btype)  # 0.5Hz ~ 3.5Hz 대역통과 필터
    ac_ppg = filtfilt(b, a, signal_detrended)
    dc_ppg = signal_detrended - ac_ppg

    # 2. 극점 검출
    peaks, _ = find_peaks(ac_ppg, distance=fs/2)
    troughs, _ = find_peaks(-ac_ppg, distance=fs/2)

    # 중복을 제거하고 순서대로 정렬
    unique_peaks_troughs = np.unique(np.concatenate((peaks, troughs)))
    t_peaks_troughs = t[unique_peaks_troughs]
    signal_peaks_troughs = ac_ppg[unique_peaks_troughs]

    # 3. 엔벨로프 생성
    upper_envelope = CubicSpline(peaks / fs, ac_ppg[peaks])
    lower_envelope = CubicSpline(troughs / fs, ac_ppg[troughs])

    # 4. 보상 가중치 곡선 계산
    fluctuation_envelope = abs(upper_envelope(t)) + abs(lower_envelope(t))
    # 엔벨로프의 표준편차의 6배에 해당하는 신호 임계값, 이 임계값 이상의 모든 값을 임계값으로 대체
    threshold = 6 * np.std(fluctuation_envelope)
    fluctuation_envelope[fluctuation_envelope > threshold] = threshold
    # 불연속성을 수정한 후, 변동 곡선의 역수에 로그를 취하면 보상 가중치(wcomp)를 얻을 수 있습
    compensation_weight = np.log1p(1 / fluctuation_envelope)

    # 5. PPG 파형 재구성, 보상 가중치 곡선을 적용함으로써, 재구성된 파형을 도출
    reconstructed_ppg = compensation_weight * ac_ppg

    # 6. 스무딩 (5점 이동 평균 필터 사용)
    window_length = 5  # 이동 평균 필터 길이
    window = np.ones(window_length) / window_length
    smoothed_signal = convolve(reconstructed_ppg, window, mode='same')

    # 7. Normalization
    normalized_signal = (smoothed_signal - np.min(smoothed_signal)) / (
            np.max(smoothed_signal) - np.min(smoothed_signal))

    if show_graph:
        # 결과 시각화
        fig, axs = plt.subplots(num="바이탈링 PPG 파형 신호 전처리 결과 그래프, (주)지티에이컴", nrows=4, ncols=1, figsize=(16, 8), sharex=True)

        plt.suptitle(filename)

        # 원본 파형 플롯
        axs[0].plot(t, signal_detrended, label='signal_detrended', color='red')
        axs[0].set_title('(a) signal_detrended waveform = filtfilt(b, a, raw_ppg)')
        axs[0].set_xlabel('Time(s)')
        axs[0].set_ylabel('Amplitude')
        axs[0].legend()

        # 새로운 Y축 생성
        ax2 = axs[0].twinx()
        ax2.plot(t, raw_ppg, label='원본 raw_ppg', color='green')
        ax2.set_ylabel('원본 raw_ppg (a.u.)')
        ax2.legend(loc='upper left')

        axs[1].plot(t, ac_ppg, label='Filtered PPG (AC)', color='red')
        axs[1].plot(t, upper_envelope(t), 'b--', label='Upper Envelope')
        axs[1].plot(t, lower_envelope(t), 'g--', label='Lower Envelope')
        axs[1].plot(t_peaks_troughs, signal_peaks_troughs, 'ro', label='Peaks & Troughs')
        axs[1].set_title('(b) Envelope (=CubicSpline(peaks / fs, ac_ppg[peaks]) and AC component (=filtfilt(b, a, signal_detrended))')
        axs[1].set_xlabel('Time(s)')
        axs[1].set_ylabel('Amplitude')
        axs[1].legend()

        # # 새로운 Y축 생성
        # ax3 = axs[1].twinx()
        # ax3.plot(t, fluctuation_envelope, label='fluctuation_envelope = abs(upper_envelope(t)) + abs(lower_envelope(t))', color='black')
        # ax3.set_ylabel('Weight')
        # ax3.legend(loc='upper right')

        axs[2].plot(t, reconstructed_ppg, label='Reconstructed PPG', color='red')
        axs[2].set_title('(c) Reconstructed PPG waveform = compensation_weight * ac_ppg')
        axs[2].set_xlabel('Time(s)')
        axs[2].set_ylabel('Amplitude')
        axs[2].legend()

        # 새로운 Y축 생성
        ax4 = axs[2].twinx()
        ax4.plot(t, compensation_weight, label='compensation_weight = np.log1p(1 / fluctuation_envelope)', color='black')
        ax4.set_ylabel('Weight')
        ax4.legend(loc='upper left')

        axs[3].plot(t, normalized_signal, label='normalized_signal', color='blue')
        axs[3].set_title('(d) normalized_signal = reconstructed_ppg 변환된 최종 PPG 파형')
        axs[3].set_xlabel('Time(s)')
        axs[3].set_ylabel('Weight')
        axs[3].legend()

        for ax in axs:
            ax.axhline(y=0, color='grey', linestyle='-')
            for x in np.arange(0, t[-1], 1):
                ax.axvline(x=x, color='grey', linestyle='--', linewidth=0.5)

        plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # 상하, 좌우 여백 최소화
        plt.tight_layout()
        plt.show()

    return normalized_signal

# ===============================================================================================
# Preprocessing : ppg의 전처리를 담당하는 부분, 주파수 샘플링에 따라 다르게 처리함
# ===============================================================================================
def process_signal(filename, s, LED):
    if LED == 'GREEN':
        s.v = -np.array(s.input_green[s.index_100:])
    elif LED == 'IR':
        s.v = -np.array(s.input_ir[s.index_100:])
    elif LED == 'RED':
        s.v = -np.array(s.input_red[s.index_100:])
    else:
        s.v = NULL

    # 큐빅 스플라인 보간을 통한 복원 곡선 생성, 복원 곡선을 적용한 파형 재구성, 장규화 과정을 거침.
    s.filt_sig = yjs_reconstruction_ppg(filename, s.v, 4, 0.5, 5.0, s.fs, btype='band')

    s.start = 0
    s.end = len(s.filt_sig)

    return s

# ===============================================================================================
# get_fs_from_filename : 파일명에 100Hz 또는 25Hz 이름이 들어간 경우, fs 값을 선택하는 함수
# ===============================================================================================
def get_fs_from_filename(filename):
    if '_100Hz' in filename:
        return 100
    elif '_25Hz' in filename:
        return 25
    else:
        return 25  # default value

def get_signal_path(data_path):
    if data_path == "" or data_path == None:
        sig_path = filedialog.askopenfilename(title='Select SIGNAL file', filetypes=[("Input Files", ".csv")])
    else:
        sig_path = data_path

    start_c = sig_path.rfind('/') + 1 if sig_path.rfind('/') > 0 else sig_path.rfind('\\')
    stop_c = sig_path.rfind('.')
    rec_name = sig_path[start_c:stop_c]
    sig_format = sig_path[len(sig_path) - sig_path[::-1].index('.'):]

    return sig_path, rec_name, sig_format

def convert_input_signals(df, fs):
    # 원 파일에서 각각의 값들을 읽어서 저장한다.
    if 'GREEN' in df.columns:                   # PC 파이썬 버전에서 실시간 데이터 수집한 경우 해당
        input_green = df['GREEN'].tolist()
        input_ir = df['IR'].tolist()
        input_red = df['RED'].tolist()
    elif 'green_led_counts' in df.columns:      # 앱에서 25, 100Hz 데이터 수집한 경우 해당
        input_green = df['green_led_counts'].tolist()
        input_ir = df['IR_led_counts'].tolist()
        input_red = df['RED_led_counts'].tolist()
    elif ' green_led_counts' in df.columns:    # 과거 실수로 앞에 공백은 둔 경우에 해당함. (무시해도 됨)
        input_green = df[' green_led_counts'].tolist()
        input_ir = df['IR_led_counts'].tolist()
        input_red = df['RED_led_counts'].tolist()
    else:
        raise KeyError("DataFrame 컬럼에 'GREEN' 또는 'green_led_counts'가 없습니다.")
        exit(0)

    if 'rr_confidence' in df.columns:
        input_rr = df['rr_confidence'].tolist()
        input_rr_interval = df['rr'].tolist()
    else:
        input_rr = []
        input_rr_interval = []

    # acc_xyz 리스트 구성
    if 'acc_x' in df.columns:
        acc_xyz = [(row['acc_x'], row['acc_y'], row['acc_z']) for _, row in df.iterrows()]
    elif 'Acc_X' in df.columns:
        acc_xyz = [(row['Acc_X'], row['Acc_Y'], row['Acc_Z']) for _, row in df.iterrows()]
    elif 'acceleration_x' in df.columns:
        acc_xyz = [(row['acceleration_x'], row['acceleration_y'], row['acceleration_z']) for _, row in df.iterrows()]
    else:
        acc_xyz = []

    # 원 파일이 100Hz 인 경우
    if fs==100 or len(input_rr) == 0:
        index_100 = 1500             # 1.5초 부터 시작 적용 (초기 압앞 신호가 찌그러지는 경우을 제외하려고 추가한 코드임)
        motion_intervals, xyz_diffs = motion_check(acc_xyz, index_100)  # 일정 구간 움직인 부분만 표시하도록 코드 수정함
        input_rr = []
        input_rr_interval = []
    # 원 파일이 25Hz 인 경우, 100Hz로 리샘플링을 진행함.
    else:
        # 100Hz로 전부 리샘플링을 진행함. 이후 모든 계산이나 그래프표시 등등 100Hz로 표시하고, 저장 파일도 100Hz로 저장함.
        input_green = interpolate_signal(input_green, 25, 100)
        input_ir = interpolate_signal(input_ir, 25, 100)
        input_red = interpolate_signal(input_red, 25, 100)

        input_rr = interpolate_signal(input_rr, 25, 100)
        input_rr_interval = interpolate_signal(input_rr_interval, 25, 100)

        # rr 신뢰도가 100% 처음 시작되는 지점의 1초전부터 신호 분석으로 사용하기 위함.
        # index_100 = np.where(input_rr >= 100)[0][0]
        indices = np.where(input_rr >= 100)[0]
        if indices.size > 0:
            index_100 = indices[0]
        else:
            # 100 이상인 값이 없을 경우의 처리
            index_100 = 0  # 또는 다른 적절한 값

        # acc_xyz 리스트를 각 축별로 분리
        acc_x = [x[0] for x in acc_xyz]
        acc_y = [x[1] for x in acc_xyz]
        acc_z = [x[2] for x in acc_xyz]

        # 각 X,Y, Z축별로 100Hz로 인터폴레이션 수행
        resampled_acc_x = interpolate_signal(acc_x, 25, 100)
        resampled_acc_y = interpolate_signal(acc_y, 25, 100)
        resampled_acc_z = interpolate_signal(acc_z, 25, 100)

        # 필요한 경우, 각 축의 데이터를 다시 튜플로 결합
        acc_xyz = list(zip(resampled_acc_x, resampled_acc_y, resampled_acc_z))

        motion_intervals, xyz_diffs = motion_check(acc_xyz, index_100)  # 일정 구간 움직인 부분만 표시하도록 코드 수정함

    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # 모든 결과값은 100Hz 기준으로 계산하도록 함.
    s = DotMap()
    s.fs = 100
    s.index_100 = index_100

    s.input_green = input_green
    s.input_ir = input_ir
    s.input_red = input_red

    s.acc_xyz = acc_xyz
    s.xyz_diffs = xyz_diffs
    s.motion_intervals = motion_intervals
    s.input_rr_interval = input_rr_interval
    s.input_rr = input_rr
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

    return s

def interpolate_signal(signal, old_fs, new_fs):
    # Interpolate a signal from old_fs to new_fs using linear interpolation.
    # Create a time array for the original signal
    old_time = np.linspace(0, len(signal) / old_fs, len(signal), endpoint=False)

    # Create a time array for the resampled signal
    new_time = np.linspace(0, len(signal) / old_fs, int(len(signal) * new_fs / old_fs), endpoint=False)

    # Use numpy's interp function to perform the linear interpolation
    new_signal = np.interp(new_time, old_time, signal)

    return new_signal

# ===============================================================================================
# ppg_convert : ppg 신호를 불어와서 신호 전처리 후에 피크 특징점을 구하고,혈압, 혈당 계산을 하는 메인 코드
# ===============================================================================================
def ppg_convert(fs, sig_path, rec_name):

    # --------------------------------------------------------------------------------
    # [1] 바이탈링 PPG 원본 csv 데이터 파일 불러오기
    df = pd.read_csv(sig_path)
    # 데이터 파일 중에 GREEN, IR 값들의 배열을 읽어 들이고, fs가 25Hz 인 경우 RR 신뢰도가 100%되는 index_100 지점을 찾음
    # --------------------------------------------------------------------------------

    yprint(f'\n\t...PPG 데이터 변환을 시작합니다.-----------------------------------------------')

    # --------------------------------------------------------------------------------
    # [2] 25Hz인 샘플링된 PPG Raw Data의 경우, 100Hz로 리샘플링(업스케일링)함,
    # index_100번째이후 신호를 역상한 결과를 반환함. 모든 계산은 100Hz 모드로 함.
    # PPG Raw Data 신호를 필터 처리하고, 스무딩 과정을 거치고, 최종적으로 0~1사이값으로 정규화 처리한 결과를 리턴함.
    converted_ppg = convert_input_signals(df, fs)
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # [3] 아래 계산은, 모두 index_100 번째 데이터부터 계산하고, 100Hz로 리샘플된 데이터로 모두 계산함.
    filtered_ppg = process_signal(sig_path, converted_ppg, LED='GREEN')
    # --------------------------------------------------------------------------------

    # --------------------------------------------------------------------------------
    # [4] 시뮬레이터 유사 포맷으로 파일 저장
    ppg_result_file_save(sig_path, filtered_ppg)
    # --------------------------------------------------------------------------------

    yprint(f'\t...원본 PPG 데이터의 필터링 및 100Hz 변환을 종료합니다..--------------------------------------------')

    return 0

def compute_xyz_diff(curr, prev):
    x1, y1, z1 = curr
    x2, y2, z2 = prev
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2 + (z1 - z2) ** 2)


def ppg_result_file_save(sig_path, ppg_data):
    index_100 = ppg_data.index_100

    # 1. 원래의 파일명 csv을 가져와서 저장할 파일명 생성
    base_filename = sig_path.split('.csv')[0]  # 파일 확장자 제거
    save_filename = base_filename + '_to_100Hz_gil.csv'

    app.output_file_entry.insert(0, save_filename)

    # 2. 저장할 csv 파일의 컬럼 헤더 및 데이터 작성
    with open(save_filename, 'w') as f:
        f.write("PPG,Type_1\n")

        for idx in range(len(ppg_data.filt_sig)):
            ppg_value = float(ppg_data.filt_sig[idx]) if isinstance(ppg_data.filt_sig[idx], str) else ppg_data.filt_sig[idx]
            f.write(f"{0},{ppg_value:.6f}\n")

    yprint(f'\t>>>> 원본 PPG 데이터를 {save_filename} 파일로 필터 변환된 결과 파일을 저장 했습니다.')

# 일정 구간 움직임 부분만 표시하도록 수정함. (YANG)
def motion_check(data, index_100=100, gap_threshold=100, min_motion_length=50, min_motion_duration=50, diff_cut = 50):
    # 모든 데이터에 대해 xyz_diffs 계산
    xyz_diffs = [0] * len(data)  # 0으로 초기화된 리스트 생성

    # index_100 번째부터 xyz_diffs 값을 계산
    for i in range(index_100 + 1, len(data)):
        xyz_diffs[i] = abs(compute_xyz_diff(data[i], data[i - 1]))

    threshold = np.percentile(xyz_diffs[index_100+1:], 75)  # index_100 이후의 값들을 사용하여 75 백분위수 계산
    # diff_cut = xyz_diff의 절대값이 50이상인 경우, 움직임으로 간주. 나중에 이값은 조정해야 함.
    # yprint(f'threshold = [{threshold}]')

    # 움직임 여부 리스트 구성
    motion_flags = [0] * len(data)
    is_moving = False
    counter = 0
    motion_start_idx = 0

    # index_100 번째부터 움직임 체크 시작
    for i in range(index_100 + 1, len(data)):
        diff = xyz_diffs[i]
        if diff > threshold and diff > diff_cut:
            if not is_moving:
                motion_start_idx = i
            counter = 0
            is_moving = True
            motion_flags[i] = 1
        elif is_moving:
            counter += 1
            motion_flags[i] = 1
            if counter > gap_threshold:     # 연속해서 100개 이상인지 체크, 순간적인 움직임은 제거하려고 함.
                if (i - motion_start_idx) >= min_motion_length:    # 최소 움직임 구간의 길이가 50개는 넘어야 함.
                    for j in range(i - counter, i):
                        motion_flags[j] = 0
                is_moving = False
                counter = 0

    # 추가적으로 길이 너무 짭은 움직임 구간은 제거함
    continuous_motion = 0
    for i in range(index_100, len(motion_flags)):
        if motion_flags[i] == 1:
            continuous_motion += 1
        else:
            if continuous_motion < min_motion_duration:    # 연속된 모션 구간이 50개 이하 구간은 무시함.
                for j in range(i - continuous_motion, i):
                    motion_flags[j] = 0
            continuous_motion = 0

    return motion_flags, xyz_diffs

# ==============================================================================================
def yprint(message, fontcolor='yellow'):
    # Insert message with a tag for its color.
    tag = "color_" + fontcolor
    app.output_text.insert(tk.END, message + "\n", tag)
    app.output_text.see(tk.END)

    # Apply the color to the tag.
    app.output_text.tag_config(tag, foreground=fontcolor)

class App:
    def __init__(self, win):
        self.win = win
        self.win.title("((주)지티에이컴, 가천대 길병원 부정맥 검출 AI 프로젝트 PPG 신호 변환용")
        self.win.geometry("1200x700")
        self.initialize_ui()

    def initialize_ui(self):
        self.label = tk.Label(self.win, text="측정된 PPG 신호를 100Hz로 변환하고, 필터 적용과 잡음을 제거하는 신호 처리 알고리즘 적용", font=("Consolas", 14))
        self.label.pack(pady=20)

        self.input_frame = tk.Frame(self.win)
        self.input_frame.pack(pady=10)

        self.open_button = tk.Button(self.input_frame, text="PPG CSV 파일 불러오기 (*.csv)", font=("Consolas", 13), command=self.open_file)
        self.open_button.pack(side=tk.LEFT)

        self.input_file_entry = tk.Entry(self.input_frame, width=60)
        self.input_file_entry.pack(side=tk.LEFT)

        self.output_frame = tk.Frame(self.win)
        self.output_frame.pack(pady=10)

        # self.ble2_button = tk.Button(self.output_frame, text="PPG 파형 분석 및 변환", font=("Consolas", 13), command=self.convert_files)
        # self.ble2_button.pack(side=tk.LEFT)

        self.output_file_entry = tk.Entry(self.output_frame, width=60)
        self.output_file_entry.pack(side=tk.LEFT)

        # 체크박스의 상태를 관리할 변수를 추가합니다
        self.check_input_var = tk.IntVar(value=1)

        # 체크박스를 추가합니다
        self.check_input = tk.Checkbutton(self.output_frame, variable=self.check_input_var, command=self.is_show_graph)
        self.check_input.pack(side=tk.LEFT)

        # 체크박스의 레이블을 추가합니다
        self.check_label = tk.Label(self.output_frame, text="파형 그래프 보기 여부")
        self.check_label.pack(side=tk.LEFT)

        # "로그 지우기" 버튼을 추가합니다
        self.clear_log_button = tk.Button(self.output_frame, text="디버창 로그 지우기", font=("Consolas", 13),
                                          command=self.clear_log)
        self.clear_log_button.pack(side=tk.LEFT)

        self.output_text_frame = tk.Frame(self.win)
        self.output_text_frame.pack(expand=True, fill=tk.BOTH, padx=10, pady=10)

        self.scrollbar = tk.Scrollbar(self.output_text_frame)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self.output_text = tk.Text(self.output_text_frame, wrap=tk.WORD, fg="white", bg="black",
                                   yscrollcommand=self.scrollbar.set,
                                   font=("Consolas", 11), cursor="xterm", insertbackground='yellow')
        self.output_text.pack(expand=True, fill=tk.BOTH)

        self.scrollbar.config(command=self.output_text.yview)

    def clear_log(self):
        """Clear the output_text widget's content."""
        self.output_text.delete(1.0, tk.END)
        # Clear the entry fields first
        self.input_file_entry.delete(0, 'end')
        self.output_file_entry.delete(0, 'end')

    def is_show_graph(self):
        global show_graph
        checkbox_status = self.check_input_var.get()
        if checkbox_status == 1:
            show_graph = True
        else:
            show_graph = False

    # def convert_file(self):
    def convert_files(self, files):
        for sig_path in files:
            rec_name, _ = os.path.splitext(os.path.basename(sig_path))
            sig_format = 'csv'  # 파일 선택 대화상자에서 이미 CSV로 필터링되었기 때문에

            self.input_file_entry.delete(0, 'end')
            self.output_file_entry.delete(0, 'end')
            self.input_file_entry.insert(0, sig_path)

            # Sampling frequency을 파일 확장자에 _100Hz.csv 또는 _25Hz.csv 표기된 기준으로 따른다.
            fs = get_fs_from_filename(os.path.basename(sig_path))

            if fs == 100:
                yprint(f'\n100Hz 샘플링 원본 파일 {sig_path}')
                ppg_convert(fs, sig_path, rec_name)
            else:
                yprint(f'\n25Hz 샘플링 원본 파일 {sig_path}')
                ppg_convert(fs, sig_path, rec_name)

    def open_file(self):
        # Clear the entry fields first
        self.input_file_entry.delete(0, 'end')
        self.output_file_entry.delete(0, 'end')

        filenames = filedialog.askopenfilenames(title="바이탈링의 추출된 PPG 데이터 파일 선택",
                                                filetypes=(("CSV files", "*.csv"), ("all files", "*.*")))
        if filenames:  # Ensure files were selected
            self.convert_files(filenames)  # Process all selected files

def main():
    global app
    win = tk.Tk()
    app = App(win)
    win.mainloop()

if __name__ == '__main__':
    main()
