"use client"

import {useEffect, useState, useRef} from "react";
import {PlyrPlayer} from "@/components/Player";
import {VideoCatalogue} from "@/components/VideoCatalogue";
import {backendUrl} from "@/const";
import {Video} from "@/types";
import {Button} from "@/components/ui/button";
import { Card, CardHeader, CardTitle, CardContent } from "@/components/ui/card";
import {Log2Controls} from "@/components/Log2Controls";


export default function DashboardPage() {
    const [videos, setVideos] = useState<Video[]>([]);
    const [source, setSource] = useState<(null | Video)[]>([null, null]);
    const [loading, setLoading] = useState(false);
    const [processingResults, setProcessingResults] = useState(null);
    const [isOne, setIsOne] = useState(true);
    const player1Ref = useRef(null);
    const player2Ref = useRef(null);
    useEffect(() => {
        const fetchVideos = async () => {
            const videoList = await (await fetch(`${backendUrl}/api/videos/`)).json() as Video[];
            setVideos(videoList);
        }
        fetchVideos();
    }, []);
    //
    const selectVideo = (video: Video) => {
        const newSelect = [...source];
        newSelect[Number(isOne)] = video;
        setSource(newSelect);
        setIsOne(!isOne);
    }
    const process = async () => {
        if (source[0] && source[1]){
            try{
                const result = await (await fetch(`${backendUrl}/api/compare-videos/?video_id1=${source[0]._id}&video_id2=${source[1]._id}`)).json();
                setProcessingResults(result)
            } catch (error) {
                setProcessingResults(error.message);
            } finally {
                setLoading(false);
            }
        }
    }
    const test = () => {
        player2Ref.current.plyr.currentTime = 10
    }
    const handleClick = (time, index) => {
        if (index === 1 ){
            player1Ref.current.plyr.currentTime = time;
        } else if (index === 2 ){
            player2Ref.current.plyr.currentTime = time;
        }
    }
    return <div>
        {source.every(a=>!!a) && <div className="lg:p-8"><Button onClick={()=>process()}>Process</Button></div>}
        { videos && <VideoCatalogue videos={videos} onSelectVideo={str => selectVideo(str)} /> }
        <Button onClick={()=>test()}>TEST</Button>
        <div style={{ display: "flex" }}>
            <div style={{width: '50vh', marginLeft: '12px'}}>

                { source[0] && <PlyrPlayer ref={player1Ref} options={{markers: {enabled: true, points: [{time: 2, label: 'pidor'}]}}}
                             source={{
                                 type: 'video',
                                 sources: [{src: backendUrl + source[0].url}]
                             }}/>}
            </div>
            <div style={{width: '50vh', marginLeft: '12px'}}>
                {source[1] && <PlyrPlayer ref={player2Ref} options={{markers: {enabled: true, points: [{time: 2, label: 'pidor'}]}}}
                            source={{
                                type: 'video',
                                sources: [{src: backendUrl + source[1].url}]
                            }}/> }
            </div>
            <Card className="mx-1 min-w1/5 shadow-xl rounded-2xl">
                <CardHeader>
                    <CardTitle className="text-xl font-semibold">Results</CardTitle>
                </CardHeader>
                <CardContent>
                    <pre className="text-sm bg-muted p-4 rounded-md">
                        {
                            loading ? "Loading..." : <code>{processingResults && <>
                                <Log2Controls key={1} handleSeek={handleClick} timestamps={processingResults.video1.timestamps} index={1}/>
                                <Log2Controls key={2} handleSeek={handleClick} timestamps={processingResults.video2.timestamps} index={2}/>

                            </> }</code>
                        }
                    </pre>
                </CardContent>
            </Card>
        </div>
    </div>
}

