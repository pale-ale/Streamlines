#pragma once

struct FilterConfig{
    double targetEnergy  = 1.0;
    double separation    = 0.04;
    double filterRadiusFactor = 1.0;
    double resolutionFactor = 1.0;
    double segmentSize   = 0.005;
    double sampleRadiusFactor = 1.0;
    double joinDistanceFactor = 1.0;
    double lengthenFactor     = 1.0;
    double sepToBlur = 6.0 / 5.0;
    constexpr double filterRadius() const {return 2.0 * filterRadiusFactor * resolutionFactor;}
    constexpr double filterSizeX()  const {return sepToBlur / filterRadiusFactor * filterRadius()/separation;}
    constexpr double sampleRadius() const {return sepToBlur  * separation * sampleRadiusFactor;}
    constexpr double joinDistance() const {return sepToBlur  * separation * joinDistanceFactor;}
    constexpr double lineStartLen() const {return separation * 2.5;}
    constexpr double deltaLen()     const {return separation * 1.125 * lengthenFactor;}
    constexpr double deltaMove()    const {return separation * 0.5;}
};

/*
@dataclass
class LPFilterConfig:
    target_energy     : float = 1.0
    separation        : float = 0.04
    sep_to_blur       : float = 6.0 / 5.0
    filter_radius     : float = 2.0
    sample_radius     : float =  .5
    line_segment_size : float =  .01
    @property
    def blur_min(self)      : return self.filter_radius
    @property
    def line_start_len(self): return self.separation  * 2.5
    @property
    def delta_len(self)     : return self.separation  * 1.125
    @property
    def delta_move(self)    : return self.separation  *  .5
    @property
    def join_dist(self)     : return self.separation  * self.sep_to_blur
    @property
    def filter_xsize(self)  : return self.sep_to_blur * self.filter_radius / self.separation
*/