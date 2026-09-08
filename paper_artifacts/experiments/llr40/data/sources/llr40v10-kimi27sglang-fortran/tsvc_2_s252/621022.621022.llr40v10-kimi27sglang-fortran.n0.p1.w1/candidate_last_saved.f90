subroutine tsvc_2_s252_fp64(a, b, c, LEN_1D) bind(c, name="tsvc_2_s252_fp64")
    use iso_c_binding, only: c_double, c_int64_t
    implicit none
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(inout) :: a(LEN_1D)
    real(c_double), intent(in) :: b(LEN_1D), c(LEN_1D)
    integer(c_int64_t), parameter :: BLOCK = 2048
    integer(c_int64_t) :: i, s, e

    if (LEN_1D <= 0) return

    ! First tile
    e = min(BLOCK, LEN_1D)
    do i = 1, e
        a(i) = b(i) * c(i)
    end do
    do i = 2, e
        a(i) = a(i) + a(i - 1)
    end do

    ! Remaining tiles
    do s = BLOCK + 1, LEN_1D, BLOCK
        e = min(s + BLOCK - 1, LEN_1D)
        do i = s, e
            a(i) = b(i) * c(i)
        end do
        a(s) = a(s) + a(s - 1)
        do i = s + 1, e
            a(i) = a(i) + a(i - 1)
        end do
    end do
end subroutine tsvc_2_s252_fp64
