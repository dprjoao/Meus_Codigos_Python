from pathlib import Path

#from tkinter import *
# Explicit imports to satisfy Flake8
from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage, filedialog, Label, Frame, Toplevel, IntVar, Radiobutton
import segyio
import numpy as np
import matplotlib.pyplot as plt
import pylops


t = None
il = None
xl = None
t = None
wav_est = None
m_tbt = None
tmin = None
tmax = None 
epsI_entry = None

OUTPUT_PATH = Path(__file__).parent
ASSETS_PATH = OUTPUT_PATH / Path(r"C:\Users\DELL\Documents\GitHub\Tkinter-Designer\build\assets\frame0")


def relative_to_assets(path: str) -> Path:
    return ASSETS_PATH / Path(path)

def load_file():
    global t, data_cube, il, xl
    filename = filedialog.askopenfilename()
    if filename:
        with segyio.open(filename, iline=189, xline=193) as stack:
                ils, xls, twt = stack.ilines, stack.xlines, stack.samples
                data_cube = segyio.cube(stack)
                n_traces = stack.tracecount 
                tr = stack.attributes(segyio.TraceField.TraceNumber)[-1]
                if not isinstance(tr, int):
                    tr = stack.attributes(segyio.TraceField.TraceNumber)[-2] + 1
                tr = int(tr[0])
        il, xl, t = ils, xls, twt
        il_start, il_end = il[0], il[-1]
        xl_start, xl_end = xl[0], xl[-1]
        dt = t[1] - t[0]

        status_label.config(text=f"File loaded: {filename}")
        # Enable the button to open the parameter input window
        button_load.config(state="normal")
        print("File loaded successfully.")
        # Define data as 2D/3D and Post-stack/Pre-stack
        if len(data_cube.shape) == 3:
            if data_cube.shape[0] != 1:
                data_type = 'Post-stack 3D'
            else:
                if n_traces > tr > 1:   
                    data_type = 'Post-stack 3D'
                else:
                    data_type = 'Post-stack 2D'
            if data_type == 'Post-stack 3D':
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                c = ax.imshow(data_cube[..., int(4000/dt)], aspect='auto', cmap='gray_r', vmin=-0.1*data_cube.max(), vmax=0.1*data_cube.max(),
                            extent=[xl_start, xl_end, il_start, il_end])
                plt.colorbar(c, ax=ax, pad=0.01)
                plt.grid(False)
                plt.show()
            else:
                pass           
            return data_cube, il, xl, t
        else:
            return None, None, None, None


# Function to get IL number from input
def get_iline_num():
    global il_number
    try:
        il_number = int(il_entry.get())
        il_label.config(text=f"Selected IL: {il_number}")

        il_start= il[0]
        xl_start, xl_end = xl[0], xl[-1]
        dt = t[1] - t[0]

        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        c = ax.imshow(data_cube[il_number - il_start, :, int(tmin/dt):int(tmax/dt)].T, aspect='auto', cmap='gray_r', vmin=-0.1*data_cube.max(), vmax=0.1*data_cube.max(),
                        extent=[xl_start, xl_end, t[int(tmax/dt)], t[int(tmin/dt)]])
        plt.colorbar(c, ax=ax, pad=0.01)
        plt.grid(False)
        plt.show()                  
        return il_number
    except ValueError:
        status_label.config(text="Error: Please enter valid IL number.")
        return None

def open_parameter_window():
    parameter_window = Toplevel(window)
    parameter_window.title("Parameter Input")

    # Create labels and entry widgets for parameter input
    tmin_label = Label(parameter_window, text="Enter time min:")
    tmin_label.grid(row=0, column=0)

    tmin_entry = Entry(parameter_window)
    tmin_entry.grid(row=0, column=1)

    tmax_label = Label(parameter_window, text="Enter max time:")
    tmax_label.grid(row=1, column=0)

    tmax_entry = Entry(parameter_window)
    tmax_entry.grid(row=1, column=1)

    estimate_button = Button(parameter_window, text="Estimate Wavelet", command=lambda: estimate_wavelet(tmin_entry=tmin_entry,tmax_entry=tmax_entry))
    estimate_button.grid(row=3,column=1)
        
    close_button = Button(parameter_window, text="Close", command= parameter_window.destroy)
    close_button.grid(row=4,column=1)
        
# Function for wavelet estimation
def estimate_wavelet(tmin_entry, tmax_entry):
    global data_cube, t, wav_est, tmin, tmax
    tmin = float(tmin_entry.get())
    tmax = float(tmax_entry.get())
    nt_wav = 16
    nfft = 2**8
    dt = t[1] - t[0]
    wav_est_fft = np.mean(np.abs(np.fft.fft(data_cube[..., int(tmin/dt):int(tmax/dt)], nfft, axis=-1)), axis=(0, 1))
    fwest = np.fft.fftfreq(nfft, d=dt/1000)
    wav_est = np.real(np.fft.ifft(wav_est_fft)[:nt_wav])
    wav_est = np.concatenate((np.flipud(wav_est[1:]), wav_est), axis=0)
    wav_est = wav_est / wav_est.max()
    # display wavelet
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))
    fig.suptitle('Statistical wavelet estimate')
    axs[0].plot(fwest[:nfft//2], wav_est_fft[:nfft//2], 'k')
    axs[0].set_title('Frequency')
    axs[1].plot(wav_est, 'k')
    axs[1].set_title('Time');
    plt.show()

def tbt_inv_param_win():
    global epsI_entry
    inv_param_win = Toplevel(window)
    inv_param_win.title("Parameter Input")

    epsI_label = Label(inv_param_win, text="Damping (epsI):")
    epsI_label.grid(row=0, column=0)
    epsI_entry = Entry(inv_param_win)
    epsI_entry.grid(row=0, column=1)


    run_inv_button = Button(inv_param_win, text="Run inversion", command=lambda: run_inversion(inv_type=inversion_type))
    run_inv_button.grid(row=1,column=1)
        
    close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
    close_button.grid(row=2,column=1)
    return epsI_entry

def spat_inv_param_win():
    global niter_sr_entry, epsI_sr_entry, epsR_sr_entry
    inv_param_win = Toplevel(window)
    inv_param_win.title("Parameter Input")

    # Create labels and entry widgets for parameter input
    niter_label = Label(inv_param_win, text="Number of iterations:")
    niter_label.grid(row=0, column=0)
    niter_sr_entry = Entry(inv_param_win)
    niter_sr_entry.grid(row=0, column=1)

    epsI_label = Label(inv_param_win, text="Damping (epsI):")
    epsI_label.grid(row=1, column=0)
    epsI_sr_entry = Entry(inv_param_win)
    epsI_sr_entry.grid(row=1, column=1)

    epsR_label = Label(inv_param_win, text="Spatial regularization (epsR):")
    epsR_label.grid(row=2, column=0)
    epsR_sr_entry = Entry(inv_param_win)
    epsR_sr_entry.grid(row=2, column=1)

    run_inv_button = Button(inv_param_win, text="Run inversion", command=lambda: run_inversion(inv_type=inversion_type))
    run_inv_button.grid(row=3,column=1)
        
    close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
    close_button.grid(row=4,column=1)
    return niter_sr_entry, epsI_sr_entry, epsR_sr_entry

def blocky_inv_param_win():
    global niter_out_b_entry, niter_in_b_entry, mu_b_entry, epsI_b_entry, epsR_b_entry, epsRL1_b_entry
    inv_param_win = Toplevel(window)
    inv_param_win.title("Parameter Input")

    # Create labels and entry widgets for parameter input
    niter_out_label = Label(inv_param_win, text="Number of outer loop iterations:")
    niter_out_label.grid(row=0, column=0)
    niter_out_b_entry = Entry(inv_param_win)
    niter_out_b_entry.grid(row=0, column=1)
    niter_in_b_label = Label(inv_param_win, text="Number of inner loop iterations:")
    niter_in_b_label.grid(row=1, column=0)
    niter_in_b_entry = Entry(inv_param_win)
    niter_in_b_entry.grid(row=1, column=1)

    niter_b_label = Label(inv_param_win, text="Number of iterations of lsqr:")
    niter_b_label.grid(row=2, column=0)
    niter_b_entry = Entry(inv_param_win)
    niter_b_entry.grid(row=2, column=1)

    mu_b_label = Label(inv_param_win, text="Data term damping:")
    mu_b_label.grid(row=3, column=0)
    mu_b_entry = Entry(inv_param_win)
    mu_b_entry.grid(row=3, column=1)

    epsI_b_label = Label(inv_param_win, text="Damping:")
    epsI_b_label.grid(row=4, column=0)
    epsI_b_entry = Entry(inv_param_win)
    epsI_b_entry.grid(row=4, column=1)

    epsR_b_label = Label(inv_param_win, text="Spatial regularization:")
    epsR_b_label.grid(row=5, column=0)
    epsR_b_entry = Entry(inv_param_win)
    epsR_b_entry.grid(row=5, column=1)

    epsRL1_b_label = Label(inv_param_win, text="Blocky regularization:")
    epsRL1_b_label.grid(row=6, column=0)
    epsRL1_b_entry = Entry(inv_param_win)
    epsRL1_b_entry.grid(row=6, column=1)

    run_inv_button = Button(inv_param_win, text="Run inversion", command=lambda: run_inversion(inv_type=inversion_type))
    run_inv_button.grid(row=7,column=1)
        
    close_button = Button(inv_param_win, text="Close", command= inv_param_win.destroy)
    close_button.grid(row=8,column=1)
    return niter_out_b_entry, niter_in_b_entry, mu_b_entry, epsI_b_entry, epsR_b_entry, epsRL1_b_entry

# Function for post-stack inversion
def tbt_inv():
    global m_tbt
    epsI = float(epsI_entry.get())
    il_start = il[0]
    xl_start, xl_end = xl[0], xl[-1]
    dt = t[1] - t[0]
    d_small = data_cube[il_number - il_start, :, int(tmin/dt):int(tmax/dt)]
    d_small = np.swapaxes(d_small, -1, 0)
        
    print("\n -------------Running trace-by-trace inversion------------- \n")
        
    m_tbt, r_tbt = pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=np.zeros_like(d_small), explicit=True,
                                                                    epsI=epsI, simultaneous=False)
    m_tbt = np.swapaxes(m_tbt, 0, -1)
    r_tbt = np.swapaxes(r_tbt, 0, -1)
    d_small = np.swapaxes(d_small, 0, -1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    c = ax.imshow(m_tbt.T, aspect='auto', cmap='seismic', vmin=-0.1*m_tbt.max(), vmax=0.1*m_tbt.max(),
                    extent=[xl_start, xl_end, t[int(tmax/dt)], t[int(tmin/dt)]])
    plt.colorbar(c, ax=ax, pad=0.01)
    plt.grid(False)
    plt.show()
    return m_tbt


# Function for post-stack inversion
def spat_inv():
    niter_sr = int(niter_sr_entry.get())
    epsI_sr = float(epsI_sr_entry.get())
    epsR_sr = float(epsR_sr_entry.get())

    il_start = il[0]
    xl_start, xl_end = xl[0], xl[-1]
    dt = t[1] - t[0]
    d_small = data_cube[il_number - il_start, :, int(tmin/dt):int(tmax/dt)]
    d_small = np.swapaxes(d_small, -1, 0)

    print("\n -------------Running spatially regularized simultaneous inversion------------- \n")
        
    if m_tbt is None:
        m0 = np.zeros_like(d_small)
    else:
        m0 = m_tbt.T
        
    m_relative_reg, r_relative_reg = \
        pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=m0, epsI=epsI_sr, epsR=epsR_sr, 
                                            **dict(iter_lim=niter_sr, show=2))

    m_relative_reg = np.swapaxes(m_relative_reg, 0, -1)
    r_relative_reg = np.swapaxes(r_relative_reg, 0, -1)
    d_small = np.swapaxes(d_small, 0, -1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    c = ax.imshow(m_relative_reg.T, aspect='auto', cmap='seismic', vmin=-0.1*m_relative_reg.max(), vmax=0.1*m_relative_reg.max(),
                extent=[xl_start, xl_end, t[int(tmax/dt)], t[int(tmin/dt)]])
    plt.colorbar(c, ax=ax, pad=0.01)
    plt.grid(False)
    plt.show()

# Function for post-stack inversion
def blocky_inv():
    niter_b = int(niter_in_b_entry.get())
    niter_out_b = int(niter_out_b_entry.get())
    niter_in_b = int(niter_in_b_entry.get())
    mu_b = float(mu_b_entry.get())
    epsI_b = float(epsI_b_entry.get())
    epsR_b = float(epsR_b_entry.get())
    epsRL1_b = float(epsRL1_b_entry.get())

    il_start = il[0]
    xl_start, xl_end = xl[0], xl[-1]
    dt = t[1] - t[0]
    d_small = data_cube[il_number - il_start, :, int(tmin/dt):int(tmax/dt)]
    d_small = np.swapaxes(d_small, -1, 0)

    print("\n -------------Running spatially regularized blocky promoting simultaneous inversion------------- \n")
        
    m_blocky, r_blocky = \
        pylops.avo.poststack.PoststackInversion(d_small, wav_est/2, m0=np.zeros_like(d_small), explicit=False, 
                                                    epsR=epsR_b, epsRL1=epsRL1_b,
                                                    **dict(mu=mu_b, niter_outer=niter_out_b, 
                                                        niter_inner=niter_in_b, show=True,
                                                        iter_lim=niter_b, damp=epsI_b))
    m_blocky = np.swapaxes(m_blocky, 0, -1)
    r_blocky = np.swapaxes(r_blocky, 0, -1)
    d_small = np.swapaxes(d_small, 0, -1)
        
    fig, ax = plt.subplots(1, 1, figsize=(10, 5))
    c = ax.imshow(m_blocky.T, aspect='auto', cmap='seismic', vmin=-0.1*m_blocky.max(), vmax=0.1*m_blocky.max(),
            extent=[xl_start, xl_end, t[int(tmax/dt)], t[int(tmin/dt)]])
    plt.colorbar(c, ax=ax, pad=0.01)
    plt.grid(False)
    plt.show()


window = Tk()

window.geometry("550x257")
window.configure(bg = "#4281A4")

canvas = Canvas(
    window,
    bg = "#4281A4",
    height = 257,
    width = 550,
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
canvas.pack(expand=True, fill="both")
canvas.place(x = 0, y = 0)

canvas.create_text(
    180.0,
    20.0,
    text="Enter IL:",
    fill="#000000",
    font=("Inter", 12 * -1)
)
canvas.pack(expand = True)

il_entry = Entry(window)
il_entry.place(
    x=159.0,
    y=30.0,
    width=116.0,
    height=24.0
)
button_image_disp = PhotoImage(
    file=relative_to_assets("button_disp.png"))
button_disp = Button(
    image=button_image_disp,
    borderwidth=0,
    highlightthickness=0,
    command=get_iline_num,
    relief="flat"
)
button_disp.place(
    x=159.0,
    y=68.0,
    width=116.0,
    height=24.0
)

button_image_load = PhotoImage(
    file=relative_to_assets("button_load.png"))
button_load = Button(
    image=button_image_load,
    borderwidth=0,
    highlightthickness=0,
    command=load_file,
    relief="flat"
)
button_load.place(
    x=15.0,
    y=13.0,
    width=116.0,
    height=24.0
)

button_image_wav = PhotoImage(
    file=relative_to_assets("button_wvl.png"))
button_wav = Button(
    image=button_image_wav,
    borderwidth=0,
    highlightthickness=0,
    command=open_parameter_window,
    relief="flat"
)
button_wav.place(
    x=15.0,
    y=52.0,
    width=116.0,
    height=24.0
)


button_image_close = PhotoImage(
    file=relative_to_assets("button_close.png"))
button_close = Button(
    image=button_image_close,
    borderwidth=0,
    highlightthickness=0,
    command=window.destroy,
    relief="flat"
)
button_close.place(
    x=15.0,
    y=94.0,
    width=116.0,
    height=24.0
)
image_logo = PhotoImage(
    file=relative_to_assets("image_1.png"))
image_1 = canvas.create_image(
    470.0,
    50.0,
    image=image_logo
)

def inversion_param_wind(inv_type):
    inversion_type = inv_type.get()
    if inversion_type == 1:
        blocky_inv_param_win()
    if inversion_type == 2:
    # Run regularized spatial inversion
        spat_inv_param_win()
    if inversion_type == 3:
    # Run regularized inversion
        tbt_inv_param_win()
           
def run_inversion(inv_type):
    inversion_type = inv_type.get()
    if inversion_type == 1:
        blocky_inv()
    elif inversion_type == 2:
    # Run regularized spatial inversion
        spat_inv()
    elif inversion_type == 3:
    # Run regularized inversion
        tbt_inv()




# Radio buttons for inversion type
inversion_type = IntVar(window)
inversion_type.set(1)  # Default to Blocky Inversion

blocky_inv_img = PhotoImage(
                file=relative_to_assets("blocky.png"))
blocky_radio = Radiobutton(window,
                        image=blocky_inv_img,
                        text="Blocky Inversion", 
                        variable=inversion_type, 
                        value=1,
                        relief="flat", 
                        borderwidth=0, 
                        highlightthickness=0)
blocky_radio.place(x=400, 
                   y=120)

reg_spt_img = PhotoImage(
                file=relative_to_assets("reg.png"))
reg_spatial_radio = Radiobutton(window,
                                image=reg_spt_img, 
                                text="Regularized Spatial Inversion", 
                                variable=inversion_type, value=2, 
                                relief="flat", 
                                borderwidth=0, 
                                highlightthickness=0)
reg_spatial_radio.place(x=400, 
                        y=150)

tbt_img = PhotoImage(
                file=relative_to_assets("tbt.png"))
reg_radio = Radiobutton(window,
                        image=tbt_img,
                        text="Trace-by-trace Inversion", 
                        variable=inversion_type, value=3, 
                        relief="flat", 
                        borderwidth=0, 
                        highlightthickness=0)
reg_radio.place(x=400, 
                y=180)


il_label = Label(window,
                bg="#4281A4"
)
il_label.place(
    x=280.0,
    y=38.0
)

status_label = Label(window,
                     bg="#4281A4")
status_label.place(x=25.0,
                    y=220)

run_button_img  = PhotoImage(
                file=relative_to_assets("run_inv.png"))
#Run inversion button
run_button = Button(window, 
                    image=run_button_img,
                    text="Open inversion params", 
                    command=lambda: inversion_param_wind(inversion_type), 
                    relief="flat", 
                    borderwidth=0, 
                    highlightthickness=0)
run_button.place(x=400, 
                 y=90, 
                 width=140, 
                 height=24)

window.resizable(False, False)
window.mainloop()
