import { useState, useEffect } from 'react';
import ConceptMapper from '../utils/concept_mapper';
export default function ListButton({ expanded, altLogits, concepts, bits, setBits, dataset }) {
    const [th, setTh] = useState(0.1);
    const [leftMargin, setLeftMargin] = useState(45);
    function getButtonClass(logit) {
        if (logit - 0.5 < th && logit - 0.5 > -th) return "button-unavailable";
        if (logit >= 0.5) return "button-positive";
        return "button-negative";
    }

    const toggleBit = (index) => {
        setBits(prev => {
            const next = [...prev];
            next[index] = prev[index] === 0 ? 1 : 0;
            return next;
        });
    };

    const mappedConcepts = ConceptMapper(dataset, concepts);
    useEffect(() => {
        setLeftMargin(expanded ? 222 : 144);
    }, [expanded]);


    return (
        <div className={`flex flex-col gap-2 `} style={{ marginLeft: `${leftMargin}px` }}>
            {bits.map((bit, idx) => {
                const buttonClass = getButtonClass(altLogits[idx] || 0);
                console.log(mappedConcepts);
                console.log(bit, idx, buttonClass);
                return (
                    <div
                        key={idx}
                        className={`w-[200px] p-2 rounded cursor-pointer ${buttonClass}`}
                        onClick={() => toggleBit(idx)}
                    >
                        {mappedConcepts[idx][bit]}                          
                    </div>
                )
            })}
        </div>
    );
}