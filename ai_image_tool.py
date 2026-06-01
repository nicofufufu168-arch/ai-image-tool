import streamlit as st
import pandas as pd
import requests
import base64
import zipfile
import time
from io import BytesIO
from datetime import datetime

st.set_page_config(page_title="AI 生图批量工具", page_icon="🎨", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Noto+Sans+SC:wght@300;400;500;700&display=swap');
html, body, [class*="css"] { font-family: 'Noto Sans SC', sans-serif; }
.stApp { background: #0a0d14; }
.hero {
    background: linear-gradient(135deg, #12182b 0%, #0f1520 100%);
    border: 1px solid #1e2d4a; border-radius: 16px;
    padding: 28px 36px; margin-bottom: 28px;
}
.hero h1 { color: #f1f5f9; font-size: 1.7rem; margin: 0; font-weight: 700; }
.hero p  { color: #64748b; margin: 8px 0 0; font-size: 0.9rem; }
.step-title {
    color: #94a3b8; font-size: 0.75rem; font-weight: 600;
    letter-spacing: 0.1em; text-transform: uppercase; margin: 24px 0 12px;
}
.prompt-preview {
    background: #0a0d14; border: 1px solid #1e293b; border-radius: 8px;
    padding: 10px 14px; color: #7dd3fc; font-size: 0.8rem;
    font-family: monospace; margin: 8px 0 12px; word-break: break-all;
}
section[data-testid="stSidebar"] { background: #0d1117; }
.stProgress > div > div { background: #6366f1; }
</style>
""", unsafe_allow_html=True)

# Session State
defaults = {
    "api_key": "", "base_url": "https://api.gptsapi.net",
    "df": None, "product_image": None, "results": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

RATIO_MAP = {
    "1:1": ("1:1", "2K"),
    "3:4": ("3:4", "2K"),
    "9:16": ("9:16", "2K"),
    "4:3": ("4:3", "2K"),
    "16:9": ("16:9", "2K"),
}

def poll_result(get_url, api_key, max_wait=120):
    """轮询查询结果，返回图片bytes"""
    headers = {"Authorization": f"Bearer {api_key}"}
    for _ in range(max_wait // 3):
        time.sleep(3)
        r = requests.get(get_url, headers=headers, timeout=30)
        r.raise_for_status()
        data = r.json().get("data", {})
        status = data.get("status", "")
        outputs = data.get("outputs", [])
        if status == "completed" and outputs:
            img_url = outputs[0] if isinstance(outputs[0], str) else outputs[0].get("url", "")
            img_r = requests.get(img_url, timeout=60)
            img_r.raise_for_status()
            return img_r.content
        elif status in ["failed", "error", "canceled"]:
            raise ValueError(f"生成失败：{data}")
    raise TimeoutError("生图超时，请重试")

def generate_text_to_image(prompt, aspect_ratio, resolution, api_key, base_url):
    """文生图"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    resp = requests.post(
        f"{base_url}/api/v3/openai/gpt-image-2/text-to-image",
        headers=headers, json=payload, timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    get_url = data.get("urls", {}).get("get") or data.get("data", {}).get("urls", {}).get("get")
    if not get_url:
        raise ValueError(f"未找到查询URL：{data}")
    return poll_result(get_url, api_key)

def generate_image_to_image(prompt, image_url, aspect_ratio, resolution, api_key, base_url):
    """图生图"""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "input_urls": [image_url],
        "aspect_ratio": aspect_ratio,
        "resolution": resolution,
    }
    resp = requests.post(
        f"{base_url}/api/v3/openai/gpt-image-2/image-edit",
        headers=headers, json=payload, timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    get_url = data.get("urls", {}).get("get") or data.get("data", {}).get("urls", {}).get("get")
    if not get_url:
        raise ValueError(f"未找到查询URL：{data}")
    return poll_result(get_url, api_key)

def upload_image_to_imgbb(image_bytes, api_key_imgbb=None):
    """上传图片到 imgbb 获取URL（免费图床）"""
    # 使用 imgbb 免费API
    encoded = base64.b64encode(image_bytes).decode()
    resp = requests.post(
        "https://api.imgbb.com/1/upload",
        data={"key": "your_imgbb_key", "image": encoded},
        timeout=30,
    )
    if resp.status_code == 200:
        return resp.json()["data"]["url"]
    return None

def load_excel(file):
    df = pd.read_excel(file)
    rename = {}
    for col in df.columns:
        c = col.strip()
        if "编号" in c or "序号" in c: rename[col] = "场景编号"
        elif "中文" in c: rename[col] = "中文场景描述"
        elif "英文" in c or "prompt" in c.lower(): rename[col] = "英文提示词"
        elif "风格" in c: rename[col] = "画面风格"
        elif "摆放" in c: rename[col] = "商品摆放"
        elif "不要" in c or "negative" in c.lower(): rename[col] = "不要出现的元素"
        elif "比例" in c or "ratio" in c.lower(): rename[col] = "图片比例"
    df.rename(columns=rename, inplace=True)
    for col in ["场景编号","中文场景描述","英文提示词","画面风格","商品摆放","不要出现的元素","图片比例"]:
        if col not in df.columns: df[col] = ""
    df["场景编号"] = df["场景编号"].astype(str).str.strip().str.zfill(2)
    return df

def build_full_prompt(row):
    parts = [str(row.get("英文提示词","")).strip()]
    style = str(row.get("画面风格","")).strip()
    pos   = str(row.get("商品摆放","")).strip()
    neg   = str(row.get("不要出现的元素","")).strip()
    if style and style != "nan": parts.append(f"Style: {style}")
    if pos   and pos   != "nan": parts.append(f"Product placement: {pos}")
    if neg   and neg   != "nan": parts.append(f"Do NOT include: {neg}")
    parts.append("Xiaohongshu aesthetic, lifestyle photography, high quality")
    return ", ".join(p for p in parts if p)

def make_zip(results):
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for scene_no, imgs in results.items():
            for i, img_bytes in enumerate(imgs, 1):
                if img_bytes:
                    zf.writestr(f"{scene_no}-{i}.jpg", img_bytes)
    buf.seek(0)
    return buf.read()

# 侧边栏
with st.sidebar:
    st.markdown("### ⚙️ API 设置")
    api_key = st.text_input("API Key", type="password", value=st.session_state.api_key, placeholder="sk-...")
    st.session_state.api_key = api_key
    base_url = st.text_input("Base URL", value=st.session_state.base_url)
    st.session_state.base_url = base_url.rstrip("/")

    if st.button("🔌 测试连接", use_container_width=True):
        if not api_key:
            st.error("请先填入 API Key")
        else:
            with st.spinner("测试中..."):
                try:
                    r = requests.get(
                        f"{st.session_state.base_url}/v1/models",
                        headers={"Authorization": f"Bearer {api_key}"},
                        timeout=10,
                    )
                    if r.status_code == 200:
                        st.success("✅ 连接成功！")
                    else:
                        st.error(f"❌ {r.status_code}")
                except Exception as e:
                    st.error(f"❌ {e}")

    st.markdown("---")
    sample = pd.DataFrame({
        "场景编号": ["01","02","03","04"],
        "中文场景描述": ["咖啡馆午后暖光","极简白色桌面","户外草地野餐","霓虹夜晚街头"],
        "英文提示词": [
            "cozy cafe afternoon, warm golden light, wooden table, soft bokeh",
            "minimalist white desk, clean aesthetic, natural light",
            "outdoor picnic on green grass, sunny day, flowers",
            "neon lights night street, rain reflections, cinematic",
        ],
        "画面风格": ["小红书","ins简约","日系清新","赛博朋克"],
        "商品摆放": ["居中","左下角","平铺","手持"],
        "不要出现的元素": ["人脸、文字","logo、人","人脸","人脸"],
        "图片比例": ["3:4","1:1","3:4","9:16"],
    })
    buf = BytesIO()
    sample.to_excel(buf, index=False)
    buf.seek(0)
    st.download_button("⬇️ 下载Excel模板", data=buf.read(),
        file_name="场景提示词模板.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        use_container_width=True)

# 主区域
st.markdown("""
<div class="hero">
  <h1>🎨 AI 生图批量工具</h1>
  <p>上传商品图 + 导入场景表 → 批量生成小红书种草图 → 预览替换 → 打包下载</p>
</div>
""", unsafe_allow_html=True)

# Step 1
st.markdown('<div class="step-title">Step 1 · 上传商品参考图（可选）</div>', unsafe_allow_html=True)
product_file = st.file_uploader("上传商品图", type=["png","jpg","jpeg","webp"], label_visibility="collapsed")
if product_file:
    st.session_state.product_image = product_file
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(product_file, width=160)
    with col2:
        st.success(f"✅ 已上传：{product_file.name}")
        st.info("💡 如需图生图，请提供商品图的公开URL（图床链接）")
        product_url = st.text_input("商品图公开URL（可选，用于图生图）", placeholder="https://...")
        if product_url:
            st.session_state["product_url"] = product_url
else:
    st.session_state["product_url"] = ""

# Step 2
st.markdown('<div class="step-title">Step 2 · 导入场景提示词表</div>', unsafe_allow_html=True)
excel_file = st.file_uploader("上传Excel", type=["xlsx","xls"], label_visibility="collapsed", key="excel")
if excel_file:
    try:
        df = load_excel(excel_file)
        st.session_state.df = df
        st.success(f"✅ 读取成功，共 {len(df)} 个场景")
        with st.expander("查看场景列表"):
            st.dataframe(df[["场景编号","中文场景描述","英文提示词","图片比例"]], use_container_width=True)
    except Exception as e:
        st.error(f"读取失败：{e}")

# Step 3
st.markdown('<div class="step-title">Step 3 · 批量生图</div>', unsafe_allow_html=True)
df = st.session_state.df
api_key = st.session_state.api_key

if df is None:
    st.info("请先上传 Excel 场景表")
elif not api_key:
    st.warning("请在左侧填入 API Key")
else:
    total = len(df)
    done = sum(1 for _, row in df.iterrows()
               if len(st.session_state.results.get(str(row["场景编号"]).zfill(2), [])) >= 4)
    c1, c2, c3 = st.columns(3)
    c1.metric("总场景数", total)
    c2.metric("已完成", f"{done}/{total}")
    c3.metric("总张数", f"{done*4}/{total*4}")

    product_url = st.session_state.get("product_url", "")
    if product_url:
        st.info("🖼️ 将使用图生图模式")
    else:
        st.info("✍️ 将使用文生图模式")

    col_qty, col_res = st.columns(2)
    with col_qty:
        num_imgs = st.selectbox("每组生成张数", [1, 2, 4, 8], index=2)
    with col_res:
        resolution_choice = st.selectbox("分辨率", ["1K", "2K", "4K"], index=0)

    if st.button(f"🚀 一键批量生成（每组{num_imgs}张）", use_container_width=True, type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        total_imgs = total * num_imgs
        done_imgs = 0

        for _, row in df.iterrows():
            scene_no = str(row["场景编号"]).zfill(2)
            prompt = build_full_prompt(row)
            ratio = str(row.get("图片比例","1:1")).strip()
            aspect_ratio, _ = RATIO_MAP.get(ratio, ("1:1", "1K"))
            resolution = resolution_choice
            imgs = []

            for i in range(num_imgs):
                status_text.markdown(f"⏳ 正在生成 **场景{scene_no}** 第 {i+1}/{num_imgs} 张...（约需15-30秒）")
                try:
                    if product_url:
                        img_bytes = generate_image_to_image(
                            prompt, product_url, aspect_ratio, resolution,
                            api_key, st.session_state.base_url
                        )
                    else:
                        img_bytes = generate_text_to_image(
                            prompt, aspect_ratio, resolution,
                            api_key, st.session_state.base_url
                        )
                    imgs.append(img_bytes)
                except Exception as e:
                    st.error(f"场景{scene_no} 第{i+1}张失败：{e}")
                    imgs.append(None)
                done_imgs += 1
                progress_bar.progress(done_imgs / total_imgs)

            st.session_state.results[scene_no] = imgs

        status_text.markdown("✅ **全部生成完成！**")

# Step 4 预览
if st.session_state.results:
    st.markdown('<div class="step-title">Step 4 · 逐张预览 & 替换</div>', unsafe_allow_html=True)
    df = st.session_state.df
    if df is not None:
        for _, row in df.iterrows():
            scene_no = str(row["场景编号"]).zfill(2)
            imgs = st.session_state.results.get(scene_no, [])
            if not imgs: continue
            scene_name = str(row.get("中文场景描述","")).strip()
            prompt = build_full_prompt(row)
            ratio = str(row.get("图片比例","1:1")).strip()
            aspect_ratio, resolution = RATIO_MAP.get(ratio, ("1:1", "2K"))
            done_count = sum(1 for img in imgs if img is not None)

            with st.expander(
                f"{'✅' if done_count==4 else '⏳'} 场景{scene_no} · {scene_name} · {done_count}/4张",
                expanded=(done_count < 4)
            ):
                st.markdown(f'<div class="prompt-preview">{prompt}</div>', unsafe_allow_html=True)
                img_cols = st.columns(4)
                for i, img_bytes in enumerate(imgs):
                    with img_cols[i]:
                        if img_bytes:
                            st.image(img_bytes, caption=f"{scene_no}-{i+1}.jpg", use_container_width=True)
                            st.download_button("⬇️", data=img_bytes,
                                file_name=f"{scene_no}-{i+1}.jpg", mime="image/jpeg",
                                key=f"dl_{scene_no}_{i}", use_container_width=True)
                        else:
                            st.markdown("❌ 失败")
                        new_prompt = st.text_area("修改提示词", value=prompt,
                            key=f"prompt_{scene_no}_{i}", height=80, label_visibility="collapsed")
                        if st.button("🔄 重新生成", key=f"regen_{scene_no}_{i}", use_container_width=True):
                            with st.spinner("生成中..."):
                                try:
                                    product_url = st.session_state.get("product_url", "")
                                    if product_url:
                                        new_img = generate_image_to_image(new_prompt, product_url, aspect_ratio, resolution, api_key, st.session_state.base_url)
                                    else:
                                        new_img = generate_text_to_image(new_prompt, aspect_ratio, resolution, api_key, st.session_state.base_url)
                                    st.session_state.results[scene_no][i] = new_img
                                    st.success("✅ 已替换！")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"失败：{e}")

# Step 5 下载
if st.session_state.results:
    st.markdown("---")
    st.markdown('<div class="step-title">Step 5 · 打包下载</div>', unsafe_allow_html=True)
    all_imgs = {k: [img for img in v if img] for k, v in st.session_state.results.items() if any(v)}
    total_ready = sum(len(v) for v in all_imgs.values())
    if total_ready > 0:
        zip_bytes = make_zip(all_imgs)
        st.download_button(
            label=f"📦 一键打包下载全部图片（{total_ready}张）.zip",
            data=zip_bytes,
            file_name=f"生图结果_{datetime.now().strftime('%Y%m%d_%H%M')}.zip",
            mime="application/zip",
            use_container_width=True, type="primary"
        )
        st.caption("文件命名：01-1.jpg / 01-2.jpg / 02-1.jpg ...")
    if st.button("🗑️ 清空重新开始", use_container_width=True):
        st.session_state.results = {}
        st.session_state.df = None
        st.session_state.product_image = None
        st.rerun()
