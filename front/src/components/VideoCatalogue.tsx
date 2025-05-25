import { Sheet, SheetTrigger,SheetTitle, SheetContent } from "@/components/ui/sheet";
import { Button } from "@/components/ui/button";
import { ScrollArea } from "@/components/ui/scroll-area"


import {Video} from "@/types";
interface VideoCatalogueProps {
    videos: Video[]
    onSelectVideo: (video: Video) => void;
}
export const VideoCatalogue = (props: VideoCatalogueProps) => {
    const {videos, onSelectVideo} = props;
    console.log(videos)
    return (<Sheet>
        <SheetTrigger asChild>
            <Button variant="outline">📁 Open Video List</Button>
        </SheetTrigger>
        <SheetContent side="left" className="w-64 p-4">
            <SheetTitle>📹 Video Files</SheetTitle>
            <ScrollArea className="h-full">
                <ul className="space-y-2">
                    {videos.map((video) => (
                        <li key={video.url}>
                            <Button
                                variant="ghost"
                                className="w-full justify-start"
                                onClick={() => onSelectVideo(video)}
                            >
                                {video.video_name}
                            </Button>
                        </li>
                    ))}
                </ul>
            </ScrollArea>
        </SheetContent>
    </Sheet>)
}