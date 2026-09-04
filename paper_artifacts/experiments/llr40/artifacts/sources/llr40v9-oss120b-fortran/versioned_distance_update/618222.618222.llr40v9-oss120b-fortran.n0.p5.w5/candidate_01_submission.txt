module versioned_distance_update_mod
  use iso_c_binding
  implicit none
contains
  subroutine versioned_distance_update_fp64(a, b, c, k, len_1d, workspace, workspace_bytes) &
    bind(C, name="versioned_distance_update_fp64")
    ! Arguments
    real(c_double), intent(inout) :: a(*)
    real(c_double), intent(in) :: b(*), c(*)
    integer(c_int64_t), value :: k, len_1d
    type(c_ptr), value :: workspace
    integer(c_int64_t), value :: workspace_bytes
    integer(c_int64_t) :: i, r
    ! Parallelize over K independent chains
    !$omp parallel default(none) shared(a,b,c,len_1d,k) private(r,i)
    !$omp do schedule(static)
    do r = 1, k
      do i = r + k, len_1d, k
        a(i) = 0.75d0 * a(i - k) + b(i) * c(i)
      end do
    end do
    !$omp end do
    !$omp end parallel
  end subroutine versioned_distance_update_fp64
end module versioned_distance_update_mod
