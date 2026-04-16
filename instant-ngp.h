#ifndef INSTANT_NGP_H
#define INSTANT_NGP_H


class InstantNGP final {
public:
    InstantNGP();
    ~InstantNGP();
    InstantNGP(const InstantNGP&)                = delete;
    InstantNGP& operator=(const InstantNGP&)     = delete;
    InstantNGP(InstantNGP&&) noexcept            = default;
    InstantNGP& operator=(InstantNGP&&) noexcept = default;
};


#endif // INSTANT_NGP_H
