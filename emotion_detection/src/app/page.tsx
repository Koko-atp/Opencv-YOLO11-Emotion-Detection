"use client";

import { useEffect, useRef, useState } from "react";
import * as ort from "onnxruntime-web";

import Image from "next/image";

import normal from "../img/vewwqeyihbaf1.jpeg"
import happymonk from "../img/images.jpg"
import angry from "../img/9e3568db34049d57de8f1d858842e886.jpg"
import shock from "../img/300px-Shocked_Black_Guy.jpg"
type CvType = any ;

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);

  const [status, setStatus] = useState<string>("ยังไม่เริ่ม");
  const [emotion, setEmotion] = useState<string>("-");
  const [conf, setConf] = useState<number>(0);

  const cvRef = useRef<CvType | null>(null);
  const faceCascadeRef = useRef<any>(null);
  const sessionRef = useRef<ort.InferenceSession | null>(null);
  const classesRef = useRef<string[] | null>(null);
  const [startloop , setstloop] = useState<boolean>(false)
  // Load OpenCV.js
  // async function loadOpenCV() {
  //   if (typeof window === "undefined") return;

  //   if ((window as any).cv) {
  //     cvRef.current = (window as any).cv;
  //     return;
  //   }

  //   await new Promise<void>((resolve, reject) => {
  //     const script = document.createElement("script");
  //     script.src = "/opencv/opencv.js";
  //     script.async = true;
  //     script.onload = () => {
  //       const cv = (window as any).cv;
  //       if (!cv) return reject(new Error("OpenCV โหลดไม่สำเร็จ"));
  //       cv["onRuntimeInitialized"] = () => {
  //         cvRef.current = cv;
  //         resolve();
  //       };
  //     };
  //     script.onerror = () => reject(new Error("โหลด opencv.js ไม่สำเร็จ"));
  //     document.body.appendChild(script);
  //   });
  // }
  async function loadOpenCV() {
  if (typeof window === "undefined") return;

  // ready แล้ว
  if ((window as any).cv?.Mat) {
    cvRef.current = (window as any).cv;
    return;
  }

  await new Promise<void>((resolve, reject) => {
    const script = document.createElement("script");
    script.src = "/opencv/opencv.js";
    script.async = true;

    script.onload = () => {
      const cv = (window as any).cv;
      if (!cv) return reject(new Error("OpenCV โหลดแล้วแต่ window.cv ไม่มีค่า"));

      const waitReady = () => {
        if ((window as any).cv?.Mat) {
          cvRef.current = (window as any).cv;
          resolve();
        } else {
          setTimeout(waitReady, 50);
        }
      };

      // บาง build มี callback บาง build พร้อมทันที
      if ("onRuntimeInitialized" in cv) {
        cv.onRuntimeInitialized = () => waitReady();
      } else {
        waitReady();
      }
    };

    script.onerror = () => reject(new Error("โหลด /opencv/opencv.js ไม่สำเร็จ"));
    document.body.appendChild(script);
  });
}


  // Load Haar cascade file into OpenCV FS
  async function loadCascade() {
    const cv = cvRef.current;
    if (!cv) throw new Error("cv ยังไม่พร้อม");

    const cascadeUrl = "/opencv/haarcascade_frontalface_default.xml";
    const res = await fetch(cascadeUrl);
    if (!res.ok) throw new Error("โหลด cascade ไม่สำเร็จ");
    const data = new Uint8Array(await res.arrayBuffer());

    // เขียนไฟล์ลง OpenCV virtual FS
    const cascadePath = "haarcascade_frontalface_default.xml";
    try {
      cv.FS_unlink(cascadePath);
    } catch {}
    cv.FS_createDataFile("/", cascadePath, data, true, false, false);

    const faceCascade = new cv.CascadeClassifier();
    const loaded = faceCascade.load(cascadePath);
    if (!loaded) throw new Error("cascade load() ไม่สำเร็จ");
    faceCascadeRef.current = faceCascade;
  }

  // 3) Load ONNX model + classes
  async function loadModel() {
    const session = await ort.InferenceSession.create(
      "/models/emotion_yolo11n_cls.onnx",
      { executionProviders: ["wasm"] }
    );
    sessionRef.current = session;

    const clsRes = await fetch("/models/classes.json");
    if (!clsRes.ok) throw new Error("โหลด classes.json ไม่สำเร็จ");
    classesRef.current = await clsRes.json();
  }

  // 4) Start camera
  async function startCamera() {
    setStatus("ขอสิทธิ์กล้อง...");
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" },
      audio: false,
    });
    if (!videoRef.current) return;
    videoRef.current.srcObject = stream;
    await videoRef.current.play();
    if(startloop === false){
      requestAnimationFrame(loop);
      setstloop(true);
    }
    setStatus("Stop Camera");
  }
  
  
  function stopCamera() {
    setStatus("กำลังหยุด....");
    if (videoRef.current && videoRef.current.srcObject) {
      const tracks = videoRef.current.srcObject.getTracks();
      tracks.forEach(track => track.stop());
      videoRef.current.srcObject = null;
      setStatus("Start Camera")
    }
}


  // 5) Preprocess face ROI -> tensor
function preprocessToTensor(faceCanvas: HTMLCanvasElement) {
  const size = 128;
 
  const tmp = document.createElement("canvas");
  tmp.width = size;
  tmp.height = size;
  const ctx = tmp.getContext("2d")!;
  ctx.drawImage(faceCanvas, 0, 0, size, size);
 
  const imgData = ctx.getImageData(0, 0, size, size).data;
  const float = new Float32Array(1 * 3 * size * size);
 
  let idx = 0;
  for (let c = 0; c < 3; c++) {
    for (let i = 0; i < size * size; i++) {
      const r = imgData[i * 4] / 255;
      const g = imgData[i * 4 + 1] / 255;
      const b = imgData[i * 4 + 2] / 255;
      float[idx++] = c === 0 ? r : c === 1 ? g : b;
    }
  }
 
  return new ort.Tensor("float32", float, [1, 3, size, size]);
}
  // 6) Softmax
  function softmax(logits: Float32Array) {
    let max = -Infinity;
    for (const v of logits) max = Math.max(max, v);
    const exps = logits.map((v) => Math.exp(v - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map((v) => v / sum);
  }

  // 7) Main loop
  async function loop() {
    try {
      const cv = cvRef.current;
      const faceCascade = faceCascadeRef.current;
      const session = sessionRef.current;
      const classes = classesRef.current;

      const video = videoRef.current;
      const canvas = canvasRef.current;
      if (!cv || !faceCascade || !session || !classes || !video || !canvas) {
        requestAnimationFrame(loop);
        return;
      }
   const ctx = canvas.getContext("2d")!;
      const displayWidth = 580;
const displayHeight = 460;
canvas.width = displayWidth;
canvas.height = displayHeight;
ctx.drawImage(video, 0, 0, displayWidth, displayHeight);


      // OpenCV: read frame
      const src = cv.imread(canvas);
      const gray = new cv.Mat();
      cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);

      const faces = new cv.RectVector();
      const msize = new cv.Size(0, 0);
      faceCascade.detectMultiScale(gray, faces, 1.1, 3, 0, msize, msize);

      // วาดกรอบ + เลือกใบหน้าที่ใหญ่สุด
      let bestRect: any = null;
      let bestArea = 0;

      for (let i = 0; i < faces.size(); i++) {
        const r = faces.get(i);
        const area = r.width * r.height;
        if (area > bestArea) {
          bestArea = area;
          bestRect = r;
        }
        ctx.strokeStyle = "lime";
        ctx.lineWidth = 1;
        ctx.strokeRect(r.x, r.y, r.width, r.height);
      }

      if (bestRect) {
        // crop face into a small canvas
        const faceCanvas = document.createElement("canvas");
        faceCanvas.width = bestRect.width;
        faceCanvas.height = bestRect.height;
        const fctx = faceCanvas.getContext("2d")!;
        fctx.drawImage(
          canvas,
          bestRect.x,
          bestRect.y,
          bestRect.width,
          bestRect.height,
          0,
          0,
          bestRect.width,
          bestRect.height
        );

        // run onnx
        const input = preprocessToTensor(faceCanvas);

        // ชื่อ input/output อาจต่างกันตามการ export
        // วิธีง่าย: ใช้ key ตัวแรกของ session.inputNames
        const feeds: Record<string, ort.Tensor> = {};
        feeds[session.inputNames[0]] = input;

        const out = await session.run(feeds);
        const outName = session.outputNames[0];
        const logits = out[outName].data as Float32Array;

        const probs = softmax(logits);
        let maxIdx = 0;
        for (let i = 1; i < probs.length; i++) {
          if (probs[i] > probs[maxIdx]) maxIdx = i;
        }

        setEmotion(classes[maxIdx] ?? `class_${maxIdx}`);
        setConf(probs[maxIdx] ?? 0);

        ctx.fillStyle = "rgba(0,0,0,1)";
        ctx.fillRect(bestRect.x, Math.max(0, bestRect.y - 28),bestRect.width , 28);
        if (bestRect.width < 120){
          ctx.fillRect(bestRect.x, Math.max(0, bestRect.y - 28),120 , 28);
        }
        ctx.fillStyle = "white";
        ctx.font = "16px sans-serif";
        ctx.fillText(
          `${classes[maxIdx]} ${(probs[maxIdx] * 100).toFixed(1)}%`,
          bestRect.x + 6,
          bestRect.y - 8,
        );
      }
      
      // cleanup
      src.delete();
      gray.delete();
      faces.delete();

      requestAnimationFrame(loop);
    } catch (e: any) {
      setStatus(`ผิดพลาด: ${e?.message ?? e}`);
    }
  }

  // Boot sequence
 useEffect(() => {
  (async () => {
    try {
      setStatus("กำลังโหลด OpenCV...");
      await loadOpenCV();

      setStatus("กำลังโหลด Haar cascade...");
      await loadCascade();

      setStatus("กำลังโหลดโมเดล ONNX...");
      await loadModel();

      setStatus("Start Camera");
    } catch (e: any) {
      setStatus(`เริ่มต้นไม่สำเร็จ: ${e?.message ?? e}`);
    }
  })();
}, []);


  return (
   <main
  className="min-h-screen p-6 bg-repeat flex items-center justify-center max-h-screen"
  style={{
    backgroundImage: "url('/bg1.jpg')",
    backgroundRepeat: "repeat",
    backgroundSize: "cover",
  }}
>
  <div className="flex flex-col items-center space-y-6">


    {/* Title */}
    <h1
      className="
        inline-block
        text-5xl
        text-black
        font-sarabun
        bg-blue-200
        border-2
        border-black
        rounded-full 
        shadow-xl
        px-4
        py-2
      "
    >
      Face Emotion ˙✧˖°📷 ⋆
    </h1>
        <div className="
        text-center
        inline-block
        text-sm
        font-extralight
        bg-blue-200
        text-amber-900
        border-2 border-black
        rounded-full
        shadow-xl
        justify-self-start
        px-3
        py-2
        transition
        duration-300
        hover:bg-blue-300
        hover:scale-105" >
          {status === "Start Camera" ?<p>
          หมายเหตุ: ต้องกดปุ่ม Start เพื่อขอสิทธิ์เปิดกล้อง <br />
          Note: Press “Start” to enable the camera
          </p>:
          <p>
            หมายเหตุ: กด Stop Camera เพื่อหยุด <br/>
            Note: Press "Stop Camera" to stopthe camera
          </p>
        }
        </div>

    {/* Status */}
    <div className="text-center bg-white/70 px-4 py-2 rounded-xl border text-black">
      <div className="text-sm">สถานะ: {status}</div>
      <div className="text-sm">
        Emotion: <b>{emotion}</b> | Conf: <b>{(conf * 100).toFixed(2)}%</b>
      </div>
    </div>

    {/* Main content: camera + reaction */}
    <div className="flex gap-12 items-start">

      {/* LEFT: Camera + Note */}
      <div className="flex flex-col  items-center space-y-1">


        <video ref={videoRef} className="hidden h-10" playsInline />

        <div className=" rounded-2xl">
          <canvas ref={canvasRef} className=" rounded-2xl h-100" />
        </div>

        <button
  onClick={status === "Start Camera"? startCamera : stopCamera}
  className="
  mt-2
  px-6 py-2
  rounded-full
  bg-white
  text-black
  border-2 border-black
  font-semibold
  shadow-lg
    transition
    duration-300
    hover:bg-green-300
    hover:scale-105
    "
>
  {status}
</button>

      </div>

      {/* RIGHT: Honest Reaction + Image */}
      <div className="flex flex-col items-center space-y-3">

        <p
          className="
            inline-block
            text-sm
            font-extralight
            bg-blue-300
            text-brown-700
            border-2 border-black
            rounded-full
            shadow-xl
            px-3
            py-1
          "
        >
          Honest Reaction ٩(◕‿◕)۶
        </p>
        <div className="max-w-full h-32 rounded-2xl conte">
        {emotion === "neutral" && <Image src={normal} className="max-w-fit h-32 rounded-2xl" alt="normal" />}
        {emotion === "surprise" && <Image src={shock} className="max-w-fit h-32 rounded-2xl" alt="surprise" />}
        {emotion === "happy" && <Image src={happymonk} className="max-w-fit h-32 rounded-2xl" alt="happy" />}
        {emotion === "angry" && <Image src={angry} className="max-w-fit h-32 rounded-2xl" alt="angry" />}
        </div>

      </div>

    </div>

  </div>
</main>
  )
}
