// app/upload/page.tsx
"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import {backendUrl} from "@/const";
import { redirect } from "next/navigation";

export default function UploadPage() {
    const [videoUrl, setVideoUrl] = useState<string | null>(null)
    const [info, setInfo] = useState<any>(null)
    const [file, setFile] = useState<File | null>(null)
    const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {

        const file = e.target.files?.[0]
        if (file) {
            setFile(file)
            const objectUrl = URL.createObjectURL(file)
            setVideoUrl(objectUrl)
        }
    }
    const handleMetadata = (e: React.SyntheticEvent<HTMLVideoElement>) => {
        const video = e.currentTarget
        setInfo({
            duration: video.duration,
            resolution: `${video.videoWidth}x${video.videoHeight}`,
            name: (file && file.name) || '?'
        })
    }
    const handleUpload = async () => {
        if (!file) return

        const formData = new FormData()
        formData.append("video_file", file)

        const res = await fetch(`${backendUrl}/api/videos/?video_name=${file.name}`, {
            method: "POST",
            body: formData,
        })

        const data = await res.json()
        redirect('/dashboard')
    }

    return (
        <div className="max-w-md mx-auto mt-10 p-4 border rounded-lg space-y-4">
            <Input
                type="file"
                accept="video/*"
                onChange={(e) => handleFileChange(e)}
            />
            <Button onClick={handleUpload} disabled={!file}>
                Upload Video
            </Button>
            {info && (
                <div className="mt-4 text-sm text-gray-700 dark:text-gray-300">
                    <p>🕒 Duration: {info.duration.toFixed(2)} seconds</p>
                    <p>📐 Resolution: {info.resolution}</p>
                    <p> Name: {info.name}</p>

                </div>
            )}            {videoUrl && (
            <video controls width={720} src={videoUrl}   onLoadedMetadata={handleMetadata} />
        )}
        </div>
    )
}
