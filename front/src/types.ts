export interface VideoTimestamp {
    event: string;
    start: number;
    end: number;
}

export interface Video {
    _id: string;
    video_name: string;
    file_path: string;
    timestamps: VideoTimestamp[];
    metadata: any;
    created_at: string;
    url: string;
}
