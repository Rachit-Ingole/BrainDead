import React from 'react';

function Navbar({ userId, setUserId, onGetRecommendations }) {
  return (
    <nav className="bg-black border-b border-gray-800 px-4 py-4 flex flex-col md:flex-row items-center justify-between">
      <div className="flex items-center mb-4 md:mb-0">
        <h1 className="text-2xl font-bold text-red-600">🎬 ReelSense</h1>
        <span className="ml-2 text-sm text-gray-400">Explainable Movie Recommender</span>
      </div>
      <div className="flex items-center space-x-4">
        <input
          type="number"
          placeholder="Enter User ID"
          value={userId}
          onChange={(e) => setUserId(e.target.value)}
          className="px-4 py-2 bg-gray-800 text-white rounded-md focus:outline-none focus:ring-2 focus:ring-red-600"
        />
        <button
          onClick={onGetRecommendations}
          className="px-6 py-2 bg-red-600 hover:bg-red-700 text-white font-semibold rounded-md transition-colors"
        >
          Get Recommendations
        </button>
      </div>
    </nav>
  );
}

export default Navbar;