import { useState, useEffect } from 'react';
import BarChart from './barchart';
export default function Intervene(
    {
        concepts,
        imageSrc,
        conceptLogits
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
        <div className="flex flex-row gap-4 items-center">
            <div>
                <img src={imageSrc} alt="Generated" style={{ maxWidth: '256px', height: 'auto' }} />
            </div>
            <div>
                <BarChart concepts={concepts} conceptLogits={conceptLogits} />
            </div>
            {/* <div>
                <svg>
                    {bits.map((bit, idx) => )}
                </svg>
                
            </div> */}

            {/* 8 toggle switches */}


        </div>
    );
}