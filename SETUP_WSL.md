# WSL2 (Windows) 환경 셋업 가이드 — Noise2NoiseFlow

새 Windows 머신의 WSL2(Ubuntu)에서 이 레포를 받아 GPU로 돌리기 위한 단계별 가이드.
각 단계 시작에 **"이미 되어있다면 SKIP"** 확인용 명령어를 같이 적어둠.

---

## 0. 전제

- Windows 11 (또는 Windows 10 21H2+)
- NVIDIA GPU (CUDA 지원)
- 관리자 권한

> **중요**: WSL2에서 GPU를 쓰려면 **Windows 호스트에만** NVIDIA 드라이버를 설치함.
> WSL 내부(Ubuntu)에는 CUDA Toolkit만 설치하고, GPU 드라이버는 설치하지 **않음**.

---

## 1. WSL2 + Ubuntu 설치

**SKIP 확인** (PowerShell):
```powershell
wsl -l -v
```
이미 `Ubuntu`가 `VERSION 2`로 뜨면 1단계 SKIP.

**설치** (PowerShell 관리자 모드):
```powershell
wsl --install -d Ubuntu-22.04
wsl --set-default-version 2
```
재부팅 후 Ubuntu를 한 번 실행해서 username/password 세팅.

---

## 2. Windows 호스트에 NVIDIA 드라이버 설치

**SKIP 확인** (WSL Ubuntu 안에서):
```bash
nvidia-smi
```
GPU가 정상 출력되면 2단계 SKIP.

**설치**:
- https://www.nvidia.com/Download/index.aspx 에서 본인 GPU용 **Game Ready** 또는 **Studio** 드라이버(최신) 설치.
- WSL용 별도 드라이버 아님 — 일반 Windows 드라이버에 WSL GPU 지원이 이미 포함됨.
- 설치 후 WSL 재시작: PowerShell에서 `wsl --shutdown` 후 다시 Ubuntu 실행.

---

## 3. Ubuntu 기본 패키지

**SKIP 확인**:
```bash
which git && which curl && which build-essential 2>/dev/null; dpkg -l | grep -E "build-essential|git|curl" | head
```
`build-essential`, `git`, `curl` 다 있으면 SKIP.

**설치**:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y build-essential git curl wget ca-certificates \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1
```
(`libgl1` 등은 `opencv-python` 런타임 의존성)

---

## 4. Miniconda 설치

**SKIP 확인**:
```bash
conda --version
```
뜨면 SKIP.

**설치**:
```bash
cd ~
curl -LO https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $HOME/miniconda3
$HOME/miniconda3/bin/conda init bash
exec bash   # 또는 터미널 재시작
```

---

## 5. 레포 클론

```bash
mkdir -p ~/research && cd ~/research
git clone <YOUR_FORK_URL> Noise2NoiseFlow
# 예: git clone https://github.com/SamsungLabs/Noise2NoiseFlow.git
cd Noise2NoiseFlow
```

---

## 6. Conda 환경 생성

**중요**: 레포의 `requirements.txt`는 torch 1.11 (Python 3.7–3.10) 기준이지만,
**현재 저자가 실제로 쓰는 환경은 Python 3.12 + torch 2.9**임. 아래는 현재 사용 버전 기준.

```bash
conda create -n n2nf python=3.12 -y
conda activate n2nf
```

---

## 7. PyTorch (CUDA 빌드) 설치

**SKIP 확인**:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```
`True`면 SKIP.

**설치** (CUDA 12.4 휠, torch 2.9):
```bash
pip install --upgrade pip
pip install torch==2.9.0 torchvision==0.24.0 --index-url https://download.pytorch.org/whl/cu124
```
GPU 드라이버가 오래돼서 CUDA 12.4를 못 받으면 `cu121` 또는 `cu118`로 대체.
확인: `nvidia-smi` 오른쪽 상단의 "CUDA Version" 이상의 런타임을 고를 수 있음.

**동작 확인**:
```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```
`True Nvidia <이름>` 나오면 OK.

---

## 8. 나머지 Python 의존성

```bash
pip install matplotlib scipy scikit-learn scikit-image \
            ipdb opencv-python h5py tensorboard
```
> `requirements.txt`는 구버전 핀이라 그대로 쓰면 torch 1.11과 충돌함. 위처럼 핀 없이 설치 권장.

---

## 9. (선택) 데이터셋 경로 세팅

`noise2noiseflow/commands.sh` 및 각 스크립트 내 `sidd_path`, `sidd_medium_path`를
본인 머신의 SIDD 데이터 위치로 수정.
(WSL에서 Windows 디스크 접근 시: `/mnt/c/...`)

---

## 10. 동작 테스트

```bash
cd noise2noiseflow
python -c "import torch, cv2, h5py, skimage, tensorboard; print('imports OK')"
python -c "import torch; print('GPU:', torch.cuda.is_available())"
```
둘 다 OK면 셋업 완료. `commands.sh`의 학습 명령 중 하나로 실제 러닝 확인.

---

## 트러블슈팅

- **`torch.cuda.is_available() == False`**
  → Windows 쪽 NVIDIA 드라이버 재설치 후 `wsl --shutdown`. WSL 안에 별도 드라이버 절대 설치 금지.
- **`libGL.so.1: cannot open shared object file`** (opencv)
  → `sudo apt install -y libgl1 libglib2.0-0`
- **`nvidia-smi: command not found` (WSL 내부)**
  → Windows 드라이버가 너무 오래됐음. 최신 Game Ready/Studio로 업데이트.
- **CUDA OOM**
  → 학습 스크립트의 batch size 축소.
