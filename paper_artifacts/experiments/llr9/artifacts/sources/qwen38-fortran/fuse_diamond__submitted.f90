subroutine fuse_diamond_fp64(a, out, LEN_1D) bind(C, name="fuse_diamond_fp64")
    use, intrinsic :: iso_c_binding
    integer(c_int64_t), value, intent(in) :: LEN_1D
    real(c_double), intent(in) :: a(*)
    real(c_double), intent(out) :: out(*)
    integer(c_int64_t) :: i
    real(c_double) :: t

    do i = 1, LEN_1D
        t = a(i) * a(i)
        out(i) = (t + 1.0_c_double) * (t - 1.0_c_double)
    end do
end subroutine fuse_diamond_fp64
