import React from "react";
import MovieCard from "./MovieCard";

function MovieGrid({ recommendations = [], loading = false }) {
  
  if (loading) {
    return (
      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 p-4">
        {Array.from({ length: 10 }).map((_, i) => (
          <MovieCard key={i} isSkeleton={true} />
        ))}
      </div>
    );
  }

  if (!recommendations || recommendations.length === 0) {
    return (
      <div className="text-center text-gray-400 text-lg mt-20 flex flex-col items-center justify-center h-64 border-2 border-dashed border-gray-700 rounded-xl mx-4">
        <span className="text-4xl mb-4">🔍</span>
        <p>No recommendations found.</p>
        <p className="text-sm text-gray-500 mt-2">Try entering a User ID like 1, 4, or 10.</p>
      </div>
    );
  }

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-6 p-4">
      {recommendations.map((movie) => (
     
        <MovieCard key={movie.movieId} movie={movie} />
      ))}
    </div>
  );
}

export default MovieGrid;