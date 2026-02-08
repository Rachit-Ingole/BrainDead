import React, { useEffect, useState } from "react";

const MovieCard = ({ movie, isSkeleton = false }) => {
  const [posterUrl, setPosterUrl] = useState(null);
  const [loading, setLoading] = useState(true);

  if (isSkeleton) {
    return (
      <div className="bg-gray-800 rounded-xl overflow-hidden shadow-lg animate-pulse flex flex-col h-96">
        <div className="w-full h-64 bg-gray-700"></div>
        <div className="p-4 space-y-3 flex-grow">
          <div className="h-5 bg-gray-700 rounded w-3/4"></div>
          <div className="h-4 bg-gray-700 rounded w-1/2"></div>
          <div className="h-16 bg-gray-700 rounded w-full mt-auto"></div>
        </div>
      </div>
    );
  }


  if (!movie) return null;


  useEffect(() => {
    let isMounted = true; 

    if (movie.tmdbId) {
      const apiKey = import.meta.env.VITE_TMDB_API_KEY; 
      
      fetch(`https://api.themoviedb.org/3/movie/${movie.tmdbId}?api_key=${apiKey}`)
        .then((res) => res.json())
        .then((data) => {
          if (isMounted && data.poster_path) {
            setPosterUrl(`https://image.tmdb.org/t/p/w500${data.poster_path}`);
          }
        })
        .catch((err) => console.error("Poster Fetch Error:", err))
        .finally(() => {
          if (isMounted) setLoading(false);
        });
    } else {
      setLoading(false);
    }

    return () => { isMounted = false; };
  }, [movie.tmdbId]);


  return (
    <div className="bg-gray-800 rounded-xl overflow-hidden shadow-lg hover:scale-105 transition-transform duration-300 flex flex-col h-full group">
      
      {/* Poster Image Layer */}
      <div className="h-64 overflow-hidden bg-gray-900 relative">
        {loading ? (
          <div className="w-full h-full flex items-center justify-center bg-gray-900 animate-pulse">
            <span className="text-gray-600 text-xs">Loading Visuals...</span>
          </div>
        ) : posterUrl ? (
          <img 
            src={posterUrl} 
            alt={movie.title} 
            className="w-full h-full object-cover" 
            loading="lazy" 
          />
        ) : (
          <div className="w-full h-full flex items-center justify-center text-gray-500 bg-gray-900">
            <span className="text-4xl">🎬</span>
          </div>
        )}
        
        {/* Match Percentage Badge */}
        <div className="absolute top-2 right-2 bg-green-600/90 backdrop-blur-sm text-white text-xs font-bold px-2 py-1 rounded-full shadow-lg border border-green-400/30 z-10">
          {movie.match_percentage}% Match
        </div>
      </div>

      {/* Content Layer */}
      <div className="p-4 flex flex-col flex-grow relative">
        <h3 className="font-bold text-white text-lg leading-tight mb-2 line-clamp-2" title={movie.title}>
          {movie.title}
        </h3>
        
        <div className="flex items-center justify-between mb-3">
          <div className="flex items-center text-yellow-400 text-sm font-mono bg-gray-900/50 px-2 py-1 rounded">
            <span className="mr-1">⭐</span>
            {movie.predicted_rating} <span className="text-gray-500 text-xs ml-1">/ 5.0</span>
          </div>
        </div>

        {/* Explainability Section */}
        <div className="mt-auto pt-3 border-t border-gray-700">
          <p className="text-gray-300 text-xs italic">
            <span className="text-blue-400 font-semibold not-italic">AI Insight: </span>
            "{movie.explanation}"
          </p>
        </div>
      </div>
    </div>
  );
};

export default MovieCard;