module tsvc_2_vpvts_mod
  implicit none
contains

  subroutine tsvc_2_vpvts_fp64(pa, pb, len_1d, s) bind(C, name="tsvc_2_vpvts_fp64")
    use, intrinsic :: iso_c_binding
    type(c_ptr), value, intent(in) :: pa
    type(c_ptr), value, intent(in) :: pb
    integer(c_int64_t), value, intent(in) :: len_1d
    integer(c_int64_t), value, intent(in) :: s
    real(c_double), pointer, contiguous :: a(:)
    real(c_double), pointer, contiguous :: b(:)
    integer(c_int64_t) :: i
    real(c_double) :: sd
    if (len_1d > 0) then
      call c_f_pointer(pa, a, shape=[len_1d])
      call c_f_pointer(pb, b, shape=[len_1d])
      sd = real(s, c_double)
      !$omp parallel do default(none) shared(a, b, sd, len_1d) schedule(static)
      do i = 1, len_1d
        a(i) = a(i) + b(i) * sd
      end do
      !$omp end parallel do
    end if
  end subroutine tsvc_2_vpvts_fp64

end module tsvc_2_vpvts_mod
