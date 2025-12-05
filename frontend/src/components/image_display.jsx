import { useState, useEffect } from 'react';
import BarChart from './barchart';
export default function ImageDisplay(
    {
        concepts,
        imageSrc,
        conceptLogits,
        mappedConcepts
    }
) {


    const toggleBit = (index) => {
        setBits(prev => {
            const next = [...prev];
            next[index] = prev[index] === 0 ? 1 : 0;
            return next;
        });
    };



    return (
        <div className="flex flex-row gap-4 items-center justify-center">
            <div>
                <img src={imageSrc} alt="Generated" style={{ maxWidth: '256px', height: 'auto' }} />
            </div>
            <div className="bg-zinc-100 p-4 rounded-lg">
                <BarChart concepts={concepts} conceptLogits={conceptLogits} mappedConcepts={mappedConcepts} />
            </div>



        </div>
    );
}