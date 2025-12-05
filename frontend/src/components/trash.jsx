{bits.map((bit, idx) => (
                    <div
                        key={idx}
                        className="flex items-center justify-between gap-4"
                    >
                        {/* {Math.abs(conceptLogits[idx] - 0.5) > th ? (
                            <div className="concept-button" >
                                <span className="text-gray-700 w-32">{concepts[idx]}</span>
                            </div>
                        ) : (
                            <div className="button-unavailable" >
                                <span className="text-gray-700 w-32">{concepts[idx]}</span>
                            </div>
                        )} */}

                        {/* <div className="button-negative">
                                <span className="text-gray-700 w-32">{concepts[idx]}</span>

                            </div> */}
                        {/* <span className="text-gray-700">{concepts[idx]}</span>

                        <label className="switch">
                            <input
                                type="checkbox"
                                checked={bit === 1}
                                onChange={() => toggleBit(idx)}
                            />
                            <span className="slider"></span>
                        </label> */}
                    </div>
                    // <div key={idx} className="flex items-center gap-4 button-positive">
                    //     <span className="text-gray-700 w-32">{concepts[idx]}</span>
                    //     {/* <button
                    //         onClick={() => toggleBit(idx)}
                    //         className={`w-12 h-6 flex items-center rounded-full p-1 duration-300 ease-in-out ${bit === 1 ? 'bg-green-500 justify-end' : 'bg-gray-300 justify-start'}`}
                    //     >
                    //         <div className="w-4 h-4 bg-white rounded-full shadow-md"></div>
                    //     </button> */}
                    // </div>
                ))}