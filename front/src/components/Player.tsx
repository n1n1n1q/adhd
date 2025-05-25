import {forwardRef, RefObject} from "react";
import {APITypes, PlyrProps, usePlyr} from "plyr-react";

import "plyr-react/plyr.css"

import dynamic from 'next/dynamic'
const Plyr = dynamic(() => import("plyr-react"), { ssr: false });

export const PlyrPlayer = forwardRef<APITypes, PlyrProps>((props, ref) => {
    return <Plyr ref={ref} {...props} />
})
export const Plyr2 = forwardRef<APITypes, PlyrProps>((props, ref) => {
    const { source, options = null, ...rest } = props
    const raptorRef = usePlyr(ref, {
        source,
        options,
    }) as RefObject<HTMLVideoElement>;
    return <video ref={raptorRef} className="plyr-react plyr" {...rest} />
})