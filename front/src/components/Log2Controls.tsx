import { Button } from "./ui/button";

export const Log2Controls = (props) =>{
    const {timestamps, handleSeek, index} = props;
    return <div className="flex flex-col gap-2">
        <h2 className="text-xl font-semibold mb-4">🎯 Timestamps Video {index}</h2>
        {timestamps.map((ts, idx) => (
            <Button
                key={idx}
                variant="default"
                className="justify-start text-left"
                onClick={() => handleSeek(ts.start, index)}
            >
                ⏱ {ts.event} ({ts.start}s)
            </Button>
        ))}
    </div>
}