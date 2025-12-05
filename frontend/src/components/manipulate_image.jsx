import { useState, useEffect } from 'react';
import DiffBarChart from './diff_barchart';
export default function AltImageDisplay(
    {
        concepts,
        imageSrc,
        conceptLogits,
        altLogits,
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
                <DiffBarChart concepts={concepts} conceptLogits={conceptLogits} altLogits={altLogits} mappedConcepts={mappedConcepts} />
            </div>


        </div>
    );
}