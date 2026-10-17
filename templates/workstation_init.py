# === Google AI Ultra 5x 云端工作站标准初始化模板 ===
# 由 colab-ultra-workstation 技能自动注入到 Colab Notebook

# --- [Cell 1] 智能硬件识别、费率监控与任务匹配顾问 ---
import platform, shutil, torch

GPU_PROFILES = {
    "CPU":          {"tier": "CPU",  "ccu_h": 0.0,   "arch": "Host CPU",          "fp8": False, "fp4": False},
    "T4":           {"tier": "T4",   "ccu_h": 1.19,  "arch": "Turing (sm_75)",    "fp8": False, "fp4": False},
    "L4":           {"tier": "L4",   "ccu_h": 1.71,  "arch": "Ada (sm_89)",       "fp8": True,  "fp4": False},
    "A100":         {"tier": "A100", "ccu_h": 5.40,  "arch": "Ampere (sm_80)",    "fp8": False, "fp4": False},
    "RTX PRO 6000": {"tier": "G4",   "ccu_h": 8.71,  "arch": "Blackwell (sm_120)","fp8": True,  "fp4": True},
    "Blackwell":    {"tier": "G4",   "ccu_h": 8.71,  "arch": "Blackwell (sm_120)","fp8": True,  "fp4": True},
    "H100":         {"tier": "H100", "ccu_h": 18.05, "arch": "Hopper (sm_90)",    "fp8": True,  "fp4": False},
}

TASK_RECOMMENDATIONS = {
    "download":    {"best": ["CPU"],        "desc": "下载模型 / 整理云盘 (0 CCU)"},
    "light_video": {"best": ["L4", "T4"],   "desc": "轻量视频/图像生成 (<20GB 显存, Wan2.1-1.3B / Flux)"},
    "heavy_video": {"best": ["G4"],         "desc": "14B 满血视频生成 (Wan2.1-14B / Hunyuan, 需 96GB G4 全驻留)"},
    "llm_small":   {"best": ["L4", "T4"],   "desc": "1.5B~9B LLM (Q4 选 T4; BF16/FP8/微调 选 L4)"},
    "llm_medium":  {"best": ["L4", "G4"],   "desc": "14B~32B LLM (14B FP8 / 32B Q4 选 L4; 32B BF16/微调 选 G4 96GB)"},
    "llm_large":   {"best": ["G4", "H100"], "desc": "70B~72B LLM (FP8 长上下文 / NVFP4 / QLoRA 首选 G4 96GB)"},
}

def inspect_hardware(task_type: str = "general"):
    if not torch.cuda.is_available():
        gpu_name, vram_gb, sm = "CPU", 0.0, (0, 0)
        profile = GPU_PROFILES["CPU"]
    else:
        gpu_name = torch.cuda.get_device_name(0)
        props = torch.cuda.get_device_properties(0)
        vram_gb = props.total_memory / (1024**3)
        sm = (props.major, props.minor)
        profile = next((v for k, v in GPU_PROFILES.items() if k != "CPU" and k.lower() in gpu_name.lower()),
                       {"tier": "GPU", "ccu_h": 5.0, "arch": f"sm_{sm[0]}{sm[1]}", "fp8": sm[0] >= 9 or sm == (8, 9), "fp4": sm[0] >= 10})
        if profile["tier"] == "A100" and vram_gb > 60:
            profile = {**profile, "tier": "A100-80G", "ccu_h": 7.52}

    ccu_h = profile["ccu_h"]
    hours_left = f"{1000 / ccu_h:.0f} 小时" if ccu_h > 0 else "♾️ 无限 (0 CCU)"
    
    hw_config = {
        "tier": profile["tier"],
        "gpu_name": gpu_name,
        "vram_gb": round(vram_gb, 1),
        "ccu_per_hour": ccu_h,
        "native_fp8": profile["fp8"],
        "native_fp4": profile["fp4"],
        "enable_cpu_offload": vram_gb < 45.0,
        "recommended_dtype": "fp8_e4m3fn / bfloat16" if profile["fp8"] else "bfloat16 / float16",
        "attn_backend": "sdpa",
    }

    print(f"🖥️ 当前运行时: [{hw_config['tier']}] {gpu_name} | 显存: {vram_gb:.1f} GB | 架构: {profile['arch']}")
    print(f"💰 算力费率  : {ccu_h} CCU/小时 (按 1,000 CCU 测算可跑: {hours_left})")
    print(f"⚙️ 自动调优  : Offload={hw_config['enable_cpu_offload']} | 推荐精度={hw_config['recommended_dtype']} | 原生FP8={profile['fp8']} | 原生FP4={profile['fp4']}")

    if task_type in TASK_RECOMMENDATIONS:
        rec = TASK_RECOMMENDATIONS[task_type]
        best_str = " / ".join(rec["best"])
        if hw_config["tier"] in rec["best"]:
            print(f"\n✅ [硬件匹配] 当前 [{hw_config['tier']}] 完美匹配任务「{rec['desc']}」！")
        elif task_type == "download" and ccu_h >= 5.0:
            print(f"\n🚨 [防烧钱警告] 当前正挂载高价显卡 [{hw_config['tier']}] ({ccu_h} CCU/h)！首次下载几十 GB 模型建议切回 CPU (0 CCU)！")
        elif ccu_h > 2.0 and task_type in ("llm_small", "light_video"):
            print(f"\n💸 [大炮打蚊子预警] 任务「{rec['desc']}」仅需 [{best_str}] (1.19~1.71 CCU/h)，当前 [{hw_config['tier']}] ({ccu_h} CCU/h) 偏贵！")
        else:
            print(f"\n💡 [选卡建议] 任务「{rec['desc']}」最优推荐为 [{best_str}]，当前为 [{hw_config['tier']}]。")
    return hw_config


# --- [Cell 2] 20TB Google Drive 零浏览器挂载检测与目录初始化 ---
import os
from pathlib import Path

def ensure_drive_mounted(mount_point: str = "/content/drive", allow_browser_fallback: bool = True):
    """
    优先复用 `colab dfs mount` (colab-dfs-ext) 的零浏览器静默挂载结果；
    若尚未挂载且允许回退，则调用 google.colab.drive.mount。
    """
    if os.path.ismount(mount_point):
        print(f"✅ 检测到 20TB Google Drive 已挂载 ({mount_point})，直接复用（零弹窗）！")
    elif allow_browser_fallback:
        print(f"ℹ️ 尚未检测到 {mount_point} 挂载（建议由 Agent 在本地执行 `colab dfs mount` 静默挂载），正在回退至浏览器授权挂载...")
        from google.colab import drive
        drive.mount(mount_point)
    else:
        raise RuntimeError(f"❌ {mount_point} 未挂载！请先在本地执行 `colab dfs mount` 完成零浏览器挂载。")

    workstation_root = Path(mount_point) / "MyDrive" / "AI_Workstation"
    dirs = {
        "models": workstation_root / "models",
        "loras": workstation_root / "loras",
        "hf_cache": workstation_root / "hf_cache",
        "outputs": workstation_root / "outputs",
    }
    for name, path in dirs.items():
        path.mkdir(parents=True, exist_ok=True)
        print(f"📁 [{name}] 就绪: {path}")

    os.environ["HF_HOME"] = str(dirs["hf_cache"])
    os.environ["HUGGINGFACE_HUB_CACHE"] = str(dirs["hf_cache"] / "hub")
    print(f"✅ HF_HOME 已永久绑定至 20TB 云盘: {os.environ['HF_HOME']}")
    return dirs
