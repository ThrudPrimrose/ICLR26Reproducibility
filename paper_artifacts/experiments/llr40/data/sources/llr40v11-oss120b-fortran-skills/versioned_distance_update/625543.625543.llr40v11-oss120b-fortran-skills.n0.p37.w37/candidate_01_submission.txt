subroutine versioned_distance_update_fp64(a, b, c, k, len_1d, workspace, workspace_size) bind(C)
  use iso_c_binding
  use omp_lib
  implicit none
  integer(c_int64_t), value, intent(in) :: k
  integer(c_int64_t), value, intent(in) :: len_1d
  type(c_ptr) :: workspace
  integer(c_int64_t), value, intent(in) :: workspace_size
  real(c_double), intent(inout) :: a(len_1d)
  real(c_double), intent(in) :: b(len_1d)
  real(c_double), intent(in) :: c(len_1d)
  integer(c_int64_t) :: offset, i

  if (k <= 1_c_int64_t) then
    ! Simple sequential loop when dependence distance is 1 or less
    do i = 2, len_1d
      a(i) = 0.75_c_double * a(i - 1) + b(i) * c(i)
    end do
  else
    ! Parallelize over the independent chains (offsets)
    !$omp parallel do default(none) shared(a,b,c,len_1d,k) private(offset,i)
    do offset = 1, k
      do i = offset + k, len_1d, k
        a(i) = 0.75_c_double * a(i - k) + b(i) * c(i)
      end do
    end do
    !$omp end parallel do
  end if
end subroutine versioned_distance_update_fp64
