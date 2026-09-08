module tsvc_2_s115_mod
  use iso_c_binding
  implicit none
contains
  subroutine tsvc_2_s115_fp64(a, aa, LEN_2D) bind(C, name="tsvc_2_s115_fp64")
    ! Arguments
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: aa(*)
    integer(c_int64_t), value :: LEN_2D
    ! Local variables
    integer(c_int64_t) :: j, i, base
    real(c_double) :: a_j

    if (LEN_2D <= 0) return

    !$omp parallel default(shared) private(j, i, base, a_j)
    do j = 0, LEN_2D-1
      a_j = a(j+1)
      base = j * LEN_2D
      !$omp do simd schedule(static)
      do i = j+1, LEN_2D-1
        a(i+1) = a(i+1) - aa(base + i + 1) * a_j
      end do
      ! implicit barrier at end of omp for
    end do
    !$omp end parallel
  end subroutine tsvc_2_s115_fp64
end module tsvc_2_s115_mod
