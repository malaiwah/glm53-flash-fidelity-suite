# Lambda gpu_1x_h100_sxm5, 2026-08-31 UTC: the instance is healthy, CUDA is not

Probed every 60 s for six minutes on one $4.29/h rental (us-south-3),
after the API reported the instance `active` and sshd accepted a
connection. `nvidia-smi` sees the card; CUDA refuses to initialise.
Raw output, unedited, from bin/fidelity/bench.wait_ready + one exec per probe.

```
rented b772a794b22645b6a1cdd2a6cb6964e2
ready
=== probe 0 at t+0s ===
--- uname
6.8.0-60-generic
--- nvidia-smi
Mon Aug 31 12:52:09 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   28C    P0             73W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
--- lspci
06:00.0 3D controller: NVIDIA Corporation Device 2330 (rev a1)
--- modules
nvidia_uvm           2121728  0
nvidia_drm            131072  0
nvidia_modeset       1724416  1 nvidia_drm
video                  77824  1 nvidia_modeset
--- torch
/usr/lib/python3/dist-packages/torch/cuda/__init__.py:174: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized (Triggered internally at ./c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
2.7.0 12.8 False 1

=== probe 1 at t+60s ===
--- uname
6.8.0-60-generic
--- nvidia-smi
Mon Aug 31 12:53:40 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   27C    P0             73W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
--- lspci
06:00.0 3D controller: NVIDIA Corporation Device 2330 (rev a1)
--- modules
nvidia_uvm           2121728  0
nvidia_drm            131072  0
nvidia_modeset       1724416  1 nvidia_drm
video                  77824  1 nvidia_modeset
--- torch
/usr/lib/python3/dist-packages/torch/cuda/__init__.py:174: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized (Triggered internally at ./c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
2.7.0 12.8 False 1

=== probe 2 at t+120s ===
--- uname
6.8.0-60-generic
--- nvidia-smi
Mon Aug 31 12:55:12 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   27C    P0             73W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
--- lspci
06:00.0 3D controller: NVIDIA Corporation Device 2330 (rev a1)
--- modules
nvidia_uvm           2121728  0
nvidia_drm            131072  0
nvidia_modeset       1724416  1 nvidia_drm
video                  77824  1 nvidia_modeset
--- torch
/usr/lib/python3/dist-packages/torch/cuda/__init__.py:174: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized (Triggered internally at ./c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
2.7.0 12.8 False 1

=== probe 3 at t+180s ===
--- uname
6.8.0-60-generic
--- nvidia-smi
Mon Aug 31 12:56:44 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   27C    P0             73W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
--- lspci
06:00.0 3D controller: NVIDIA Corporation Device 2330 (rev a1)
--- modules
nvidia_uvm           2121728  0
nvidia_drm            131072  0
nvidia_modeset       1724416  1 nvidia_drm
video                  77824  1 nvidia_modeset
--- torch
/usr/lib/python3/dist-packages/torch/cuda/__init__.py:174: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized (Triggered internally at ./c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
2.7.0 12.8 False 1

=== probe 4 at t+240s ===
--- uname
6.8.0-60-generic
--- nvidia-smi
Mon Aug 31 12:58:15 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   27C    P0             73W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
--- lspci
06:00.0 3D controller: NVIDIA Corporation Device 2330 (rev a1)
--- modules
nvidia_uvm           2121728  0
nvidia_drm            131072  0
nvidia_modeset       1724416  1 nvidia_drm
video                  77824  1 nvidia_modeset
--- torch
/usr/lib/python3/dist-packages/torch/cuda/__init__.py:174: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized (Triggered internally at ./c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
2.7.0 12.8 False 1

=== probe 5 at t+300s ===
--- uname
6.8.0-60-generic
--- nvidia-smi
Mon Aug 31 12:59:47 2026
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.148.08             Driver Version: 570.148.08     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA H100 80GB HBM3          On  |   00000000:06:00.0 Off |                    0 |
| N/A   28C    P0             73W /  700W |       0MiB /  81559MiB |      0%      Default |
|                                         |                        |             Disabled |
+-----------------------------------------+------------------------+----------------------+
--- lspci
06:00.0 3D controller: NVIDIA Corporation Device 2330 (rev a1)
--- modules
nvidia_uvm           2121728  0
nvidia_drm            131072  0
nvidia_modeset       1724416  1 nvidia_drm
video                  77824  1 nvidia_modeset
--- torch
/usr/lib/python3/dist-packages/torch/cuda/__init__.py:174: UserWarning: CUDA initialization: Unexpected error from cudaGetDeviceCount(). Did you run some cuda functions before calling NumCudaDevices() that might have already set an error? Error 802: system not yet initialized (Triggered internally at ./c10/cuda/CUDAFunctions.cpp:109.)
  return torch._C._cuda_getDeviceCount() > 0
2.7.0 12.8 False 1

{'terminated': 'b772a794b22645b6a1cdd2a6cb6964e2'}
destroyed b772a794b22645b6a1cdd2a6cb6964e2
```

## A fourth rental of the same type was fine

Twenty minutes later, `gpu_1x_h100_sxm5` `a4a7bb2a...` answered
`torch.cuda.is_available() == True` on the first probe, with
`nvidia-fabricmanager` **inactive and disabled** and
`nvidia-smi -q` reporting `Fabric State: Completed`. So the 802 is a
PER-HOST condition on this instance type, not a property of the type,
and it is not simply a stopped Fabric Manager: starting the service on
that healthy box in fact FAILED (exit 3) while CUDA already worked.

```
rented a4a7bb2a501944f4bd22419a3b00b3a7
ready
=== BEFORE ===
CUDA_OK True 1

=== fabricmanager unit state ===
inactive
disabled
ii  nvidia-fabricmanager-570                         570.148.08-0lambda0.22.04.1                  amd64        NVIDIA Fabric Manager for NVSwitch systems
        GPU Fabric GUID                   : N/A
    Performance State                     : P0
    Fabric
        State                             : Completed

=== start it ===
destroyed a4a7bb2a501944f4bd22419a3b00b3a7
Traceback (most recent call last):
  File "<scratch>/diag_fm.py", line 22, in <module>
    print(p.exec_stdout(mid,
  File "<repo>/bin/fidelity/sshbase.py", line 114, in exec_stdout
    return str(self.exec(machine_id, command, timeout=timeout,
  File "<repo>/bin/fidelity/sshbase.py", line 108, in exec
    raise JLError("remote command exited %s: %s"
fidelity.jlapi.JLError: remote command exited 3: Job for nvidia-fabricmanager.service failed because the control process exited with error code.
See "systemctl status nvidia-fabricmanager.service" and "journalctl -xeu nvidia-fabricmanager.service" for details.
failed
```
