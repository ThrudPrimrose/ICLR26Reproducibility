module tsvc_2_vag_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_vag_fp64(a, b, ip, len_1d) bind(C, name="tsvc_2_vag_fp64")
    use iso_c_binding
    implicit none
    integer(c_int64_t), value, intent(in) :: len_1d
    real(c_double), intent(inout) :: a(len_1d)
    real(c_double), intent(in) :: b(len_1d)
    integer(c_int32_t), intent(in) :: ip(len_1d)
    integer(c_int64_t) :: i
    !$omp parallel
    !$omp do schedule(static)
    do i = 1, len_1d
      a(i) = b(ip(i))
    end do
    !$omp end parallel
  end subroutine tsvc_2_vag_fp64
end module tsvc_2_vag_mod
