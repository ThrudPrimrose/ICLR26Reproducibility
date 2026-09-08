module tsvc_2_s252_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s252_fp64(a, b, c, len_1d) bind(C, name="tsvc_2_s252_fp64")
    ! Arguments
    integer(c_int64_t), value :: len_1d
    real(c_double), dimension(*), intent(out) :: a
    real(c_double), dimension(*), intent(in) :: b, c
    ! Local variables
    integer(c_int64_t) :: i
    ! Compute first element separately
    if (len_1d >= 1_c_int64_t) then
       a(1) = b(1) * c(1)
    end if
    ! Compute remaining elements: a(i) = b(i)*c(i) + b(i-1)*c(i-1)
    if (len_1d > 1_c_int64_t) then
      !$omp parallel do simd schedule(static)
      do i = 2_c_int64_t, len_1d
        a(i) = b(i) * c(i) + b(i-1) * c(i-1)
      end do
    end if
  end subroutine tsvc_2_s252_fp64
end module tsvc_2_s252_mod
