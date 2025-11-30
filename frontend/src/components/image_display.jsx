import { useState } from 'react';
export default function ImageDisplay() {
    const [setName, setSetName] = useState('cbae_stygan2_thr90-celebahq');
    // const [filename, setFilename] = useState('00000000.png');
    const [bits, setBits] = useState(Array(8).fill(0));
    const filename = bits.map(bit => bit.toString()).join('') + '.png';

    const [attribute1, setAttribute1] = useState(false);
    const path = `/CE_data/${setName}/${filename}`;

    const toggleBit = (index) => {
        setBits(prev => {
            const next = [...prev];
            next[index] = prev[index] === 0 ? 1 : 0;
            return next;
        });
    };

    return (
        <div className="flex flex-col gap-4 items-center">
            <img src={path} alt-text="Displayed Image" style={{ maxWidth: '100%', height: 'auto' }} />
            <p className="text-sm text-gray-600">{filename}</p>

            {/* 8 toggle switches */}
            <div className="flex flex-col gap-2">
                {bits.map((bit, idx) => (
                    <div key={idx} className="flex items-center justify-between">
                        <span>Attribute {idx+1}</span>

                        <label className="switch">
                            <input 
                                type="checkbox" 
                                checked={bit === 1}
                                onChange={() => toggleBit(idx)}
                            />
                            <span className="slider"></span>
                        </label>
                    </div>
                ))}
            </div>
            {/* <form>
                <div className="switch-container justify-between">
                    <span className='attr-text'>Attribute 1</span>
                    <label class="switch">
                        <input type="checkbox" checked={attribute1} onChange={() => setAttribute1(!attribute1)} />
                        
                        <span class="slider"></span>
                    </label>
                </div>
            </form> */}
            
        </div>            
    );
}